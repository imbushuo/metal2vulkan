// Authored half stage-boundary regression; not derived from a captured shader.
#include <metal_stdlib>
using namespace metal;

struct HalfStageInput {
    float4 position [[position]];
    half4 color [[user(color), flat]];
};

fragment half4 fragment_half_stage_io(HalfStageInput input [[stage_in]]) {
    return input.color.zzww - half4(0.125h);
}
