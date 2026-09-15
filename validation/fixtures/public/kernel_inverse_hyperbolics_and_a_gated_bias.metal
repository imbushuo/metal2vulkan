// Five `air.*` symbols the corpus reaches once or four times each and that no authored case
// covered: `air.fast_acosh.f32`, `air.fast_asinh.f32`, `air.fast_atanh.f32`, `air.fast_tan.v3f32`
// and `air.normalize_function_constant_predicate.i16`.
//
// The four transcendentals are asked only at ENDPOINTS, for the same reason
// `kernel_atan2_on_the_axes` is: SPIRV-Cross leaves these unqualified and MoltenVK answers them
// with the precise variant, while the Metal oracle runs the `fast::` one the AIR names.
// `fast::tan` and `precise::tan` disagree on more than half of ordinary arguments, so an ordinary
// argument here would be recording a MoltenVK limitation rather than a translation.
//
// The endpoints are NOT the ones IEEE-754 would give, and that is the point of the rows. Every
// value below is device-measured on Apple M3 Max under the default (fast-math) compile, and
// MoltenVK reproduces all thirteen bit for bit:
//
//   acosh(1) = +0            acosh(+inf) = +inf
//   asinh(+0) = +0           asinh(-0) = +0        <- the sign of zero is LOST
//   asinh(-inf) = NaN                              <- not -inf
//   atanh(+0) = +0           atanh(-0) = +0        <- lost here too
//   atanh(+1) = +inf         atanh(-1) = -inf
//   tan(+0) = +0             tan(-0) = -0          <- and PRESERVED here
//
// So `fast::asinh` and `fast::atanh` are not odd functions at zero even though the functions they
// approximate are, while `fast::tan` is; and `fast::asinh(-inf)` is NaN, which is what
// `log(x + sqrt(x*x + 1))` gives when the two infinities cancel. An authored case cannot assume
// "the endpoints are safe" for a `fast_` member -- it has to measure them.
//
// The one non-endpoint row is `tan(2^-20)`. The series is `x + x^3/3 + ...`, so the cubic term is
// 2^-62 -- forty-two binary places below the last bit of a binary32 near 2^-20 -- and the answer is
// exactly the argument in any implementation that is even approximately right. Metal agrees.
//
// `has_bias` is a `short` function constant that gates the `bias` argument. That is the spelling
// that makes Metal emit `air.normalize_function_constant_predicate.i16` into `air.static_init`:
// the constant is normalized to 0 or 1 before anything reads it. The three cases on this fixture
// set it to 0, 1 and 2, and the 1 and 2 cases produce byte-identical output -- which is the whole
// content of the word "normalize".
#include <metal_stdlib>
using namespace metal;

constant short has_bias [[function_constant(0)]];

kernel void kernel_inverse_hyperbolics_and_a_gated_bias(
    device const float* fin [[buffer(0)]],
    device const float3* vin [[buffer(1)]],
    device const float* bias [[buffer(2), function_constant(has_bias)]],
    device uint* out [[buffer(3)]])
{
    // air.fast_acosh.f32 -- acosh(1) is +0 and acosh(+inf) is +inf in every implementation.
    out[0] = as_type<uint>(acosh(fin[0]));
    out[1] = as_type<uint>(acosh(fin[1]));

    // air.fast_asinh.f32 -- odd, so both signed zeros pass through, and -inf stays -inf.
    out[2] = as_type<uint>(asinh(fin[2]));
    out[3] = as_type<uint>(asinh(fin[3]));
    out[4] = as_type<uint>(asinh(fin[4]));

    // air.fast_atanh.f32 -- both signed zeros, and the two poles at +-1.
    out[5] = as_type<uint>(atanh(fin[5]));
    out[6] = as_type<uint>(atanh(fin[6]));
    out[7] = as_type<uint>(atanh(fin[7]));
    out[8] = as_type<uint>(atanh(fin[8]));

    // air.fast_tan.v3f32 -- the signed zeros, and an argument so small that the cubic term of the
    // series is 42 binary places below the result's own last bit, so tan(x) == x exactly.
    float3 t = tan(vin[0]);
    out[9] = as_type<uint>(t.x);
    out[10] = as_type<uint>(t.y);
    out[11] = as_type<uint>(t.z);

    // air.normalize_function_constant_predicate.i16 -- the `short` function constant that gates the
    // `bias` argument. Metal normalizes it to 0 or 1 in `air.static_init` before anything reads it.
    float v = fin[9];
    if (has_bias) { v += bias[0]; }
    out[12] = as_type<uint>(v);
}
