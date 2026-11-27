//! Native D3D12 pipeline-library ownership and serialization.
//!
//! D3D12 retains the pointer passed to `CreatePipelineLibrary`, so a reopened library must own
//! its serialized bytes for at least as long as the COM object.

use windows::core::{HSTRING, Interface, Result};
use windows::Win32::Graphics::Direct3D12::*;

pub(crate) struct NativeLibrary {
    library: ID3D12PipelineLibrary1,
    _serialized: Vec<u8>,
}

impl NativeLibrary {
    pub(crate) fn supported(device: &ID3D12Device) -> bool {
        let mut feature = D3D12_FEATURE_DATA_SHADER_CACHE::default();
        unsafe {
            device.CheckFeatureSupport(
                D3D12_FEATURE_SHADER_CACHE,
                (&mut feature as *mut D3D12_FEATURE_DATA_SHADER_CACHE).cast(),
                std::mem::size_of_val(&feature) as u32,
            )
        }
        .is_ok()
            && feature.SupportFlags.0 & D3D12_SHADER_CACHE_SUPPORT_LIBRARY.0 != 0
    }

    pub(crate) fn empty(device: &ID3D12Device) -> Result<Self> {
        Self::open(device, Vec::new())
    }

    pub(crate) fn open(device: &ID3D12Device, serialized: Vec<u8>) -> Result<Self> {
        let device: ID3D12Device1 = device.cast()?;
        let library = unsafe { device.CreatePipelineLibrary::<ID3D12PipelineLibrary1>(&serialized)? };
        Ok(Self { library, _serialized: serialized })
    }

    pub(crate) fn store(&self, name: &str, pipeline: &ID3D12PipelineState) -> Result<()> {
        unsafe { self.library.StorePipeline(&HSTRING::from(name), pipeline) }
    }

    pub(crate) fn load_compute(
        &self,
        name: &str,
        desc: &D3D12_COMPUTE_PIPELINE_STATE_DESC,
    ) -> Result<ID3D12PipelineState> {
        unsafe { self.library.LoadComputePipeline(&HSTRING::from(name), desc) }
    }

    pub(crate) fn load_graphics(
        &self,
        name: &str,
        desc: &D3D12_GRAPHICS_PIPELINE_STATE_DESC,
    ) -> Result<ID3D12PipelineState> {
        unsafe { self.library.LoadGraphicsPipeline(&HSTRING::from(name), desc) }
    }

    pub(crate) fn load_stream(
        &self,
        name: &str,
        desc: &D3D12_PIPELINE_STATE_STREAM_DESC,
    ) -> Result<ID3D12PipelineState> {
        unsafe { self.library.LoadPipeline(&HSTRING::from(name), desc) }
    }

    pub(crate) fn serialize(&self) -> Result<Vec<u8>> {
        let mut bytes = vec![0; unsafe { self.library.GetSerializedSize() }];
        unsafe { self.library.Serialize(&mut bytes)? };
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::NativeLibrary;
    use crate::gpu::recomp;
    use crate::shaders::Pipelines;
    use std::mem::ManuallyDrop;
    use std::time::Instant;
    use windows::core::Interface;
    use windows::Win32::Graphics::Dxgi::Common::*;
    use windows::Win32::Graphics::Direct3D12::*;

    fn fixture() -> Option<(Pipelines, Vec<u8>, ID3D12RootSignature)> {
        if let Err(reason) = recomp::dxc::location() {
            recomp::runner::skip_test(&format!("no DXC: {reason}"));
            return None;
        }
        let pipelines = match Pipelines::new() {
            Ok(pipelines) => pipelines,
            Err(reason) => {
                recomp::runner::skip_test(&format!("no D3D12 device: {reason}"));
                return None;
            }
        };
        if !NativeLibrary::supported(&pipelines.dev) {
            recomp::runner::skip_test("adapter does not support D3D12 pipeline libraries");
            return None;
        }
        let source = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer output : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() { output.Store(0, 42); }
"#;
        let dxil = recomp::dxc::compile(source, "main", "cs_6_6", &[]).expect("DXC compile");
        let root = unsafe { pipelines.dev.CreateRootSignature(0, &dxil) }.expect("root signature");
        Some((pipelines, dxil, root))
    }

    fn compute_desc(dxil: &[u8], root: &ID3D12RootSignature) -> D3D12_COMPUTE_PIPELINE_STATE_DESC {
        D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: ManuallyDrop::new(Some(root.clone())),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: dxil.as_ptr().cast(),
                BytecodeLength: dxil.len(),
            },
            ..Default::default()
        }
    }

    fn drop_desc(desc: D3D12_COMPUTE_PIPELINE_STATE_DESC) {
        drop(ManuallyDrop::into_inner(desc.pRootSignature));
    }

    #[test]
    fn compute_pipeline_round_trips_through_serialized_library() {
        let Some((pipelines, dxil, root)) = fixture() else { return };
        let desc = compute_desc(&dxil, &root);
        let pso = unsafe { pipelines.dev.CreateComputePipelineState::<ID3D12PipelineState>(&desc) }
            .expect("compute pipeline");
        let library = NativeLibrary::empty(&pipelines.dev).expect("empty pipeline library");
        for n in 1..=256 {
            library.store(&format!("compute-{n:04}"), &pso).expect("store pipeline alias");
        }
        assert!(library.store("compute-0001", &pso).is_err(), "duplicate names must be rejected");

        let serialize_at = Instant::now();
        let bytes = library.serialize().expect("serialize pipeline library");
        let serialize_time = serialize_at.elapsed();
        assert!(!bytes.is_empty());

        let open_at = Instant::now();
        let reopened = NativeLibrary::open(&pipelines.dev, bytes.clone()).expect("reopen pipeline library");
        let open_time = open_at.elapsed();
        let loaded = reopened.load_compute("compute-0001", &desc).expect("load compute pipeline");
        assert!(unsafe { loaded.GetCachedBlob() }.is_ok());
        println!(
            "pipeline library: aliases=256 serialized_bytes={} serialize_us={} open_us={}",
            bytes.len(),
            serialize_time.as_micros(),
            open_time.as_micros()
        );
        drop_desc(desc);
    }

    #[test]
    fn truncated_pipeline_library_is_rejected() {
        let Some((pipelines, dxil, root)) = fixture() else { return };
        let desc = compute_desc(&dxil, &root);
        let pso = unsafe { pipelines.dev.CreateComputePipelineState::<ID3D12PipelineState>(&desc) }
            .expect("compute pipeline");
        let library = NativeLibrary::empty(&pipelines.dev).expect("empty pipeline library");
        library.store("compute-0001", &pso).expect("store pipeline");
        let mut bytes = library.serialize().expect("serialize pipeline library");
        bytes.truncate(bytes.len() / 2);
        assert!(NativeLibrary::open(&pipelines.dev, bytes).is_err());
        drop_desc(desc);
    }

    #[test]
    fn graphics_pipeline_round_trips_through_serialized_library() {
        let Some((pipelines, _, _)) = fixture() else { return };
        let source = r#"
#define RS ""
struct V { float4 p : SV_Position; };
[RootSignature(RS)] V vs(uint id : SV_VertexID) {
    V v; v.p = float4(id == 1 ? 1.0 : -1.0, id == 2 ? 1.0 : -1.0, 0.0, 1.0); return v;
}
[RootSignature(RS)] float4 ps() : SV_Target { return float4(1, 0, 1, 1); }
"#;
        let vs = recomp::dxc::compile(source, "vs", "vs_6_6", &[]).expect("compile VS");
        let ps = recomp::dxc::compile(source, "ps", "ps_6_6", &[]).expect("compile PS");
        let root = unsafe { pipelines.dev.CreateRootSignature(0, &vs) }.expect("graphics root signature");
        let mut blend = D3D12_BLEND_DESC::default();
        blend.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
        let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: ManuallyDrop::new(Some(root)),
            VS: D3D12_SHADER_BYTECODE { pShaderBytecode: vs.as_ptr().cast(), BytecodeLength: vs.len() },
            PS: D3D12_SHADER_BYTECODE { pShaderBytecode: ps.as_ptr().cast(), BytecodeLength: ps.len() },
            BlendState: blend,
            SampleMask: u32::MAX,
            RasterizerState: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_NONE,
                DepthClipEnable: true.into(),
                ..Default::default()
            },
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        let pso = unsafe { pipelines.dev.CreateGraphicsPipelineState::<ID3D12PipelineState>(&desc) }
            .expect("graphics pipeline");
        let library = NativeLibrary::empty(&pipelines.dev).expect("empty pipeline library");
        library.store("graphics-0001", &pso).expect("store graphics pipeline");
        let bytes = library.serialize().expect("serialize graphics library");
        let reopened = NativeLibrary::open(&pipelines.dev, bytes).expect("reopen graphics library");
        reopened.load_graphics("graphics-0001", &desc).expect("load graphics pipeline");
        drop(ManuallyDrop::into_inner(desc.pRootSignature));
    }

    #[repr(C, align(8))]
    struct Sub<T> {
        ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE,
        value: T,
    }

    #[repr(C)]
    struct MeshStream {
        root: Sub<*mut std::ffi::c_void>,
        mesh: Sub<D3D12_SHADER_BYTECODE>,
        topology: Sub<D3D12_PRIMITIVE_TOPOLOGY_TYPE>,
    }

    #[test]
    fn mesh_pipeline_round_trips_through_serialized_library() {
        let Some((pipelines, _, _)) = fixture() else { return };
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
        if unsafe {
            pipelines.dev.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS7,
                (&mut options as *mut D3D12_FEATURE_DATA_D3D12_OPTIONS7).cast(),
                std::mem::size_of_val(&options) as u32,
            )
        }
        .is_err()
            || options.MeshShaderTier.0 < D3D12_MESH_SHADER_TIER_1.0
        {
            recomp::runner::skip_test("adapter does not support mesh shaders");
            return;
        }
        let source = r#"
#define RS ""
struct V { float4 p : SV_Position; };
[RootSignature(RS)]
[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void main(out vertices V vertices[3], out indices uint3 triangles[1]) {
    SetMeshOutputCounts(3, 1);
    vertices[0].p = float4(-1, -1, 0, 1);
    vertices[1].p = float4( 0,  1, 0, 1);
    vertices[2].p = float4( 1, -1, 0, 1);
    triangles[0] = uint3(0, 1, 2);
}
"#;
        let mesh = recomp::dxc::compile(source, "main", "ms_6_6", &[]).expect("compile MS");
        let root: ID3D12RootSignature = unsafe { pipelines.dev.CreateRootSignature(0, &mesh) }
            .expect("mesh root signature");
        let mut stream = MeshStream {
            root: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, value: root.as_raw() },
            mesh: Sub {
                ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS,
                value: D3D12_SHADER_BYTECODE { pShaderBytecode: mesh.as_ptr().cast(), BytecodeLength: mesh.len() },
            },
            topology: Sub {
                ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY,
                value: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            },
        };
        let desc = D3D12_PIPELINE_STATE_STREAM_DESC {
            SizeInBytes: std::mem::size_of::<MeshStream>(),
            pPipelineStateSubobjectStream: (&mut stream as *mut MeshStream).cast(),
        };
        let device: ID3D12Device2 = pipelines.dev.cast().expect("ID3D12Device2");
        let pso = unsafe { device.CreatePipelineState::<ID3D12PipelineState>(&desc) }.expect("mesh pipeline");
        let library = NativeLibrary::empty(&pipelines.dev).expect("empty pipeline library");
        library.store("mesh-0001", &pso).expect("store mesh pipeline");
        let bytes = library.serialize().expect("serialize mesh library");
        let reopened = NativeLibrary::open(&pipelines.dev, bytes).expect("reopen mesh library");
        reopened.load_stream("mesh-0001", &desc).expect("load mesh pipeline");
    }
}
