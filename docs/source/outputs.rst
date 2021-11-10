Outputs and Functions
=====================


Functions
^^^^^^^^^^^^^^^^^^^^

- B_contour
- tangent_R_spline
- tangent_cylindrical
- tangent_phi_spline
- tangent_z_spline
- binormal_z_spline
- binormal_R_spline
- convert_to_spline
- binormal_phi_spline
- normal_R_spline
- normal_phi_spline
- normal_z_spline
- nu_spline
- Z0_func
- R0_func
- from_paper
- get_boundary
- grad_B_colon_grad_B
- grad_B_tensor
- grad_B_tensor_cartesian
- grad_B_tensor_cylindrical
- get_dofs
- plot
- plot_axis
- plot_boundary
- B_fieldline
- B_mag

Outputs
^^^^^^^

At first order

Scalars

- B0
- Bbar
- sG
- sigma0
- spsi
- G0
- abs_G0_over_B0
- axis_length
- etabar

Arrays

- sigma
- torsion
- varphi
- rc
- rs
- zc
- zs
- Bfield_cartesian
- Bfield_cylindrical
- X1c
- X1c_untwisted
- X1s
- X1s_untwisted
- Y1c
- Y1c_untwisted
- Y1s
- Y1s_untwisted
- Z0
- Z0p
- Z0pp
- Z0ppp
- d2_X1c_d_varphi2
- d2_Y1c_d_varphi2
- d2_Y1s_d_varphi2
- d_X1c_d_varphi
- d_X1s_d_varphi
- d_Y1c_d_varphi
- d_Y1s_d_varphi
- d_curvature_d_varphi
- d_d_phi
- d_d_varphi
- d_l_d_phi
- d_l_d_varphi
- d_phi
- d_torsion_d_varphi
- d_varphi_d_phi
- elongation
- etabar_squared_over_curvature_squared
- binormal_cylindrical
- curvature
- L_grad_B
- R0
- R0p
- R0pp
- R0ppp
- helicity
- init_axis
- inv_L_grad_B
- iota
- iotaN
- lasym
- max_elongation
- mean_elongation
- min_L_grad_B
- min_R0
- min_R0_penalty
- min_R0_threshold
- names
- nfourier
- nfp
- normal_cylindrical
- nphi
- order
- phi

At second order:

Scalars

- B2c
- B2s
- G2
- I2
- N_helicity
- beta_1s

Arrays

- B20
- B20_anomaly
- B20_mean
- B20_residual
- B20_variation
- DGeod_times_r2
- DMerc_times_r2
- DWell_times_r2
- L_grad_grad_B
- V1
- V2
- V3
- X20
- X20_untwisted
- X2c
- X2c_untwisted
- X2s
- X2s_untwisted
- Y20
- Y20_untwisted
- Y2c
- Y2c_untwisted
- Y2s
- Y2s_untwisted
- Z20
- Z20_untwisted
- Z2c
- Z2c_untwisted
- Z2s
- Z2s_untwisted
- calculate_grad_grad_B_tensor
- d2_volume_d_psi2
- d_X20_d_varphi
- d_X2c_d_varphi
- d_X2s_d_varphi
- d_Y20_d_varphi
- d_Y2c_d_varphi
- d_Y2s_d_varphi
- d_Z20_d_varphi
- d_Z2c_d_varphi
- d_Z2s_d_varphi
- grad_grad_B
- grad_grad_B_inverse_scale_length
- grad_grad_B_inverse_scale_length_vs_varphi
- grad_grad_B_tensor_cartesian
- grad_grad_B_tensor_cylindrical
- inv_r_singularity_vs_varphi
- p2
- r_singularity
- r_singularity_basic_vs_varphi
- r_singularity_residual_sqnorm
- r_singularity_theta_vs_varphi
- r_singularity_vs_varphi

At third order:

Arrays

- B0_order_a_squared_to_cancel
- X3c1
- X3c1_untwisted
- X3c3
- X3c3_untwisted
- X3s1
- X3s1_untwisted
- X3s3
- X3s3_untwisted
- Y3c1
- Y3c1_untwisted
- Y3c3
- Y3c3_untwisted
- Y3s1
- Y3s1_untwisted
- Y3s3
- Y3s3_untwisted
- Z3c1
- Z3c1_untwisted
- Z3c3
- Z3c3_untwisted
- Z3s1
- Z3s1_untwisted
- Z3s3
- Z3s3_untwisted
- d_Y3c1_d_varphi
- d_Y3s1_d_varphi
- d_X3c1_d_varphi
- flux_constraint_coefficient