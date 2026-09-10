; Authored semantic counterpart of fragment_half_stage_io.metal.
; Portable half4 varying, arithmetic, and color output through float32 stage transport.
source_filename = "fragment_half_stage_io.metal"

define <4 x half> @fragment_half_stage_io(<4 x float> %position, <4 x half> %color) {
  %selected = shufflevector <4 x half> %color, <4 x half> poison, <4 x i32> <i32 2, i32 2, i32 3, i32 3>
  %result = fsub <4 x half> %selected, <half 0xH3000, half 0xH3000, half 0xH3000, half 0xH3000>
  ret <4 x half> %result
}

!air.fragment = !{!0}
!0 = !{ptr @fragment_half_stage_io, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"half4", !"air.arg_name", !"color"}
!3 = !{!4, !5}
!4 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"position", !"air.arg_unused"}
!5 = !{i32 1, !"air.fragment_input", !"user(color)", !"air.flat", !"air.arg_type_name", !"half4", !"air.arg_name", !"color"}
