use super::TransformOptions;
use crate::reflect::RuntimeSamplerState;

impl TransformOptions {
    /// Exact operational-cache encoding of every translation option.
    /// Floating-point sampler fields retain their bits, including signed zero.
    pub fn cache_identity(&self) -> Result<Vec<u8>, String> {
        let TransformOptions {
            descriptor_layout,
            kernel_local_size,
            kernel_dispatch,
            denorm_flush_to_zero_f32,
            raster_sample_count,
            runtime_sampler_states,
            runtime_storage_image_states,
            vertex_amplification_count,
            threadgroup_memory_lengths,
        } = self;
        let samplers = runtime_sampler_states
            .iter()
            .map(|state| {
                state.map(
                    |RuntimeSamplerState {
                         min_filter,
                         mag_filter,
                         mip_filter,
                         address_mode_s,
                         address_mode_t,
                         address_mode_r,
                         coordinates,
                         compare_function,
                         max_anisotropy,
                         lod_min_clamp,
                         lod_max_clamp,
                         border_color,
                         reduction,
                         lod_bias,
                     }| {
                        (
                            min_filter,
                            mag_filter,
                            mip_filter,
                            address_mode_s,
                            address_mode_t,
                            address_mode_r,
                            coordinates,
                            compare_function,
                            max_anisotropy,
                            lod_min_clamp.to_bits(),
                            lod_max_clamp.to_bits(),
                            border_color,
                            reduction,
                            lod_bias.to_bits(),
                        )
                    },
                )
            })
            .collect::<Vec<_>>();
        serde_json::to_vec(&(
            descriptor_layout,
            kernel_local_size,
            kernel_dispatch,
            denorm_flush_to_zero_f32,
            raster_sample_count,
            samplers,
            runtime_storage_image_states.as_slice(),
            vertex_amplification_count,
            threadgroup_memory_lengths.as_slice(),
        ))
        .map_err(|error| format!("encode translation options: {error}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reflect::*;

    #[test]
    fn option_identity_covers_dispatch_layout_memory_and_float_bits() {
        let original = TransformOptions::default();
        let key = original.cache_identity().unwrap();
        for changed in [
            TransformOptions {
                kernel_local_size: [32, 1, 1],
                ..original
            },
            TransformOptions {
                kernel_dispatch: Some(KernelDispatch::Workgroups),
                ..original
            },
            TransformOptions {
                raster_sample_count: Some(4),
                ..original
            },
            TransformOptions {
                vertex_amplification_count: 2,
                ..original
            },
            TransformOptions {
                denorm_flush_to_zero_f32: true,
                ..original
            },
        ] {
            assert_ne!(key, changed.cache_identity().unwrap());
        }
        let mut changed = original;
        changed.descriptor_layout.set += 1;
        assert_ne!(key, changed.cache_identity().unwrap());
        changed = original;
        changed.threadgroup_memory_lengths[0] = Some(64);
        assert_ne!(key, changed.cache_identity().unwrap());
        changed.runtime_sampler_states[0] = Some(RuntimeSamplerState {
            min_filter: SamplerFilter::Nearest,
            mag_filter: SamplerFilter::Nearest,
            mip_filter: SamplerMipFilter::None,
            address_mode_s: SamplerAddressMode::ClampToEdge,
            address_mode_t: SamplerAddressMode::ClampToEdge,
            address_mode_r: SamplerAddressMode::ClampToEdge,
            coordinates: SamplerCoordinates::Normalized,
            compare_function: SamplerCompareFunction::None,
            max_anisotropy: 1,
            lod_min_clamp: 0.0,
            lod_max_clamp: 1.0,
            border_color: SamplerBorderColor::TransparentBlack,
            reduction: SamplerReduction::WeightedAverage,
            lod_bias: 0.0,
        });
        let positive_zero = changed.cache_identity().unwrap();
        changed.runtime_sampler_states[0].as_mut().unwrap().lod_bias = -0.0;
        assert_ne!(positive_zero, changed.cache_identity().unwrap());
    }
}
