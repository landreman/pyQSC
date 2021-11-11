Outputs and Functions
=====================

List of main outputs provided by ``pyQsc`` and functions that
can be used. Some of the functions have links to their
definitions in the code in order to get more details.

Functions
^^^^^^^^^^

- :py:mod:`from_paper <qsc.Qsc.from_paper>`: create a stellarator from a known configuration.
- :py:mod:`get_boundary <qsc.Qsc.get_boundary>`: obtain the :math:`(x,y,z)` arrays of a boundary shape at a specific radius.
- :py:mod:`plot <qsc.Qsc.plot>`: Generate a matplotlib figure with an array of plots, showing the toroidally varying properties of the configuration.
- :py:mod:`plot_axis <qsc.Qsc.plot_axis>`: Plot axis shape and the Frenet-Serret frame along the axis (optional).
- :py:mod:`plot_boundary <qsc.Qsc.plot_boundary>`: Plot the boundary of the near-axis configuration.
- :py:mod:`B_contour <qsc.Qsc.B_contour>`: Plot contours of B in the :math:`(\vartheta,\varphi)` plane with :math:`(\vartheta,\varphi)` Boozer angles
- :py:mod:`B_fieldline <qsc.Qsc.B_fieldline>`: Plot the modulus of the magnetic field B along a field line.
- :py:mod:`B_mag <qsc.Qsc.B_mag>`: Calculate the modulus of the magnetic field for a given set of coordinates :math:`(r,\theta,\phi)`
- :py:mod:`Bfield_cartesian <qsc.Qsc.Bfield_cartesian>`: magnetic field vector in cartesian coordinates
- :py:mod:`Bfield_cylindrical <qsc.Qsc.Bfield_cylindrical>`:  magnetic field vector in cylindrical coordinates
- :py:mod:`grad_B_tensor_cartesian <qsc.Qsc.grad_B_tensor_cartesian>`: grad B tensor in cartesian coordinates
- :py:mod:`grad_B_tensor_cylindrical <qsc.Qsc.grad_B_tensor_cylindrical>`: grad B tensor in cylindrical coordinates
- :py:mod:`grad_grad_B_tensor_cylindrical <qsc.Qsc.grad_grad_B_tensor_cylindrical>`: grad grad B tensor in cartesian coordinates
- :py:mod:`grad_grad_B_tensor_cartesian <qsc.Qsc.grad_grad_B_tensor_cartesian>`: grad grad B tensor in cartesian coordinates

Outputs
^^^^^^^

Scalars

- :py:mod:`iota <qsc.Qsc.iota>`: rotational transform on-axis (iota)
- :py:mod:`iotaN <qsc.Qsc.iotaN>`: iota-N where N is the helicity of the axis
- :py:mod:`axis_length <qsc.Qsc.axis_length>`: total length of the magnetic axis
- :py:mod:`Bbar <qsc.Qsc.Bbar>`: magnetic field normalization :math:`=s_\psi \times B_0`
- :py:mod:`G0 <qsc.Qsc.G0>`: lowest order Boozer function :math:`G_0`
- :py:mod:`G2 <qsc.Qsc.G2>`: higher order Boozer function :math:`G_0`
- :py:mod:`beta_1s <qsc.Qsc.beta_1s>`: Boozer function :math:`\beta_{1s}`
- :py:mod:`lasym <qsc.Qsc.lasym>`: true if stellarator-asymmetric, false otherwise
- :py:mod:`max_elongation <qsc.Qsc.max_elongation>`: maximum elongation of the first order solution
- :py:mod:`mean_elongation <qsc.Qsc.mean_elongation>`: mean elongation of the first order solution
- :py:mod:`min_R0 <qsc.Qsc.min_R0>`: minimum radial location of the axis

Arrays

- :py:mod:`sigma <qsc.Qsc.sigma>`: sigma function related to the first-order solution
- :py:mod:`torsion <qsc.Qsc.torsion>`: torsion of the axis
- :py:mod:`curvature <qsc.Qsc.curvature>`: curvature of the axis
- :py:mod:`varphi <qsc.Qsc.varphi>`: toroidal Boozer angle
- :py:mod:`phi <qsc.Qsc.phi>`: toroidal cylindrical angle
- :py:mod:`d_l_d_phi <qsc.Qsc.d_l_d_phi>`: derivative of the arclength with respect to phi
- :py:mod:`d_l_d_varphi <qsc.Qsc.d_l_d_varphi>`: derivative of the arclength with respect to varphi
- :py:mod:`elongation <qsc.Qsc.elongation>`: elongation of the first order elliptical shape
- :py:mod:`B20 <qsc.Qsc.B20>`: solution of the second order magnetic field B20
- :py:mod:`DGeod_times_r2 <qsc.Qsc.DGeod_times_r2>`: DGeod term of the Mercier criterion times r2
- :py:mod:`DMerc_times_r2 <qsc.Qsc.DMerc_times_r2>`: DMerc term of the Mercier criterion times r2
- :py:mod:`DWell_times_r2 <qsc.Qsc.DWell_times_r2>`: DWell term of the Mercier criterion times r2
- :py:mod:`d2_volume_d_psi2 <qsc.Qsc.d2_volume_d_psi2>`: Magnetic well
- :py:mod:`r_singularity <qsc.Qsc.r_singularity>`: proxy for the maximum acceptable radius


SIMSOPT related
^^^^^^^^^^^^^^^
- :py:mod:`get_dofs <qsc.Qsc.get_dofs>`: list of degrees of freedom of a particular stellarator in a format ready to be used by SIMSOPT.
- :py:mod:`set_dofs <qsc.Qsc.set_dofs>`: modify the degress of freedom of a particular stellarator resulting in a new configuration.
- :py:mod:`names <qsc.Qsc.names>`: names of degrees of freedom


Position Vector
^^^^^^^^^^^^^^^

- :py:mod:`Z0 <qsc.Qsc.Z0>`
- :py:mod:`R0 <qsc.Qsc.R0>`
- :py:mod:`X1c <qsc.Qsc.X1c>`
- :py:mod:`X1s <qsc.Qsc.X1s>`
- :py:mod:`Y1c <qsc.Qsc.Y1c>`
- :py:mod:`Y1s <qsc.Qsc.Y1s>`
- :py:mod:`X20 <qsc.Qsc.X20>`
- :py:mod:`X2c <qsc.Qsc.X2c>`
- :py:mod:`X2s <qsc.Qsc.X2s>`
- :py:mod:`Y20 <qsc.Qsc.Y20>`
- :py:mod:`Y2c <qsc.Qsc.Y2c>`
- :py:mod:`Y2s <qsc.Qsc.Y2s>`
- :py:mod:`Z20 <qsc.Qsc.Z20>`
- :py:mod:`Z2c <qsc.Qsc.Z2c>`
- :py:mod:`Z2s <qsc.Qsc.Z2s>`
- :py:mod:`X3c1 <qsc.Qsc.X3c1>`
- :py:mod:`X3c3 <qsc.Qsc.X3c3>`
- :py:mod:`X3s1 <qsc.Qsc.X3s1>`
- :py:mod:`X3s3 <qsc.Qsc.X3s3>`
- :py:mod:`Y3c1 <qsc.Qsc.Y3c1>`
- :py:mod:`Y3c3 <qsc.Qsc.Y3c3>`
- :py:mod:`Y3s1 <qsc.Qsc.Y3s1>`
- :py:mod:`Y3s3 <qsc.Qsc.Y3s3>`
- :py:mod:`Z3c1 <qsc.Qsc.Z3c1>`
- :py:mod:`Z3c3 <qsc.Qsc.Z3c3>`
- :py:mod:`Z3s1 <qsc.Qsc.Z3s1>`
- :py:mod:`Z3s3 <qsc.Qsc.Z3s3>`


Grad B
^^^^^^

- :py:mod:`grad_B_tensor_cylindrical <qsc.Qsc.grad_B_tensor_cylindrical>`
- :py:mod:`grad_B_colon_grad_B <qsc.Qsc.grad_B_colon_grad_B>`
- :py:mod:`L_grad_B <qsc.Qsc.L_grad_B>`
- :py:mod:`L_grad_grad_B <qsc.Qsc.L_grad_grad_B>`
- :py:mod:`inv_L_grad_B <qsc.Qsc.inv_L_grad_B>`
- :py:mod:`min_L_grad_Bmin_L_grad_B>`
- :py:mod:`grad_grad_B_inverse_scale_length_vs_varphi <qsc.Qsc.grad_grad_B_inverse_scale_length_vs_varphi>`
- :py:mod:`L_grad_grad_B <qsc.Qsc.L_grad_grad_B>`
- :py:mod:`grad_grad_B_inverse_scale_length <qsc.Qsc.grad_grad_B_inverse_scale_length>`
- :py:mod:`grad_grad_B <qsc.Qsc.grad_grad_B>`
