open_project ./project_1.xpr

# Clean and rebuild runs so updated constraints are picked up.
reset_run synth_1
reset_run impl_1

launch_runs synth_1 -jobs 2
wait_on_run synth_1

launch_runs impl_1 -to_step route_design -jobs 2
wait_on_run impl_1

open_run impl_1
report_timing_summary -max_paths 10 -routable_nets -report_unconstrained -warn_on_violation -file ./project_1.runs/impl_1/post_fix_timing_summary.rpt
report_io -file ./project_1.runs/impl_1/post_fix_io.rpt
report_drc -file ./project_1.runs/impl_1/post_fix_drc.rpt
close_project
exit
