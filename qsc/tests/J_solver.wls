#!/usr/bin/env wolframscript
(* ::Package:: *)

(* ::Subsection:: *)
(*Inputs*)


B00=ToExpression[$ScriptCommandLine[[2]]]
B01=ToExpression[$ScriptCommandLine[[3]]]
B1c0=ToExpression[$ScriptCommandLine[[4]]]
B1c1=ToExpression[$ScriptCommandLine[[5]]]
B1s1=ToExpression[$ScriptCommandLine[[6]]]
r=ToExpression[$ScriptCommandLine[[7]]]
alpha=ToExpression[$ScriptCommandLine[[8]]]
lambda=ToExpression[$ScriptCommandLine[[9]]]
iotaN=ToExpression[$ScriptCommandLine[[10]]]
NFP=ToExpression[$ScriptCommandLine[[11]]]
G0=ToExpression[$ScriptCommandLine[[12]]]
I2=ToExpression[$ScriptCommandLine[[13]]]


(* ::Subsection:: *)
(*Magnetic Field*)


(*Magnetic Field Strength*)
B[x_]=1.B00+1.B01 Cos[NFP x]+1.r((B1c0+B1c1 Cos[NFP x])Cos[alpha+iotaN x]+(B1s1 Sin[NFP x])Sin[alpha+iotaN x]);

(* ::Subsection:: *)
(*Integration function*)


GaussLegendreQuadrature[f_,{x_,a_,b_},n_Integer: 10,prec_: MachinePrecision]:=Module[{nodes,weights},{nodes,weights}=Most[NIntegrate`GaussRuleData[n,prec]];
(b-a) weights.Map[Function[x,f],Rescale[nodes,{0,1},{a,b}]]];


(* ::Subsection:: *)
(*Function to Find Multiple Roots of Magnetic Well*)


Options@FindRoots=Sort@Join[Options@FindRoot,{MaxRecursion->Automatic,PerformanceGoal:>$PerformanceGoal,PlotPoints->Automatic,Debug->False,ZeroTolerance->10^-9}];

FindRoots[fun_,{var_,min_,max_},opts:OptionsPattern[]]:=Module[{PlotRules,RootRules,g,g2,pts,pts2,lpts,F,sol},(*Extract the Options*)PlotRules=Sequence@@FilterRules[Join[{opts},Options@FindRoots],Options@Plot];
RootRules=Sequence@@FilterRules[Join[{opts},Options@FindRoots],Options@FindRoot];
(*Plot the function and "mesh" the point with y-coordinate 0*)g=Normal@Plot[fun,{var,min,max},MeshFunctions->(#2&),Mesh->{{0}},Method->Automatic,Evaluate@PlotRules,PlotLabel->"1-\[Lambda] B(\[Phi])",AxesLabel->{"\[Phi]"}];
(*Get the meshes zeros*)pts=Cases[g,Point[p_]:>SetPrecision[p[[1]],OptionValue@WorkingPrecision],Infinity];
(*Get all plot points*)lpts=Join@@Cases[g,Line[p_]:>SetPrecision[p,OptionValue@WorkingPrecision],Infinity];
(*Derive the interpolated data to find other zeros*)F=Interpolation[lpts,InterpolationOrder->2];
g2=Normal@Plot[Evaluate@D[F@var,var],{var,min,max},MeshFunctions->(#2&),Mesh->{{0}},Method->Automatic,Evaluate@PlotRules];
(*Get the meshes zeros and retain only small ones*)pts2=Cases[g2,Point[p_]:>SetPrecision[p[[1]],OptionValue@WorkingPrecision],Infinity];
pts2=Select[pts2,Abs[F@#]<OptionValue@ZeroTolerance&];
pts=Join[pts,pts2];(*Join all zeros*)(*Refine zeros by passing each point through FindRoot*)If[Length@pts>0,pts=Map[FindRoot[fun,{var,#},Evaluate@RootRules]&,pts];
sol=Union@Select[pts,min<=Last@Last@#<=max&];
sol,{}]]


(* ::Subsection:: *)
(*Get Roots of v_parallel*)


roots={0,0};
xmin=0;xmax=12\[Pi];
vpar[x_?NumericQ]:=1-lambda B[x];
rootsTemp=x/.FindRoots[vpar[x],{x,xmin,xmax}]//Quiet;
If[
Length[rootsTemp]<=1,0,
If[vpar[1.0001rootsTemp[[1]]]>0,
roots[[1]]=rootsTemp[[1]];If[vpar[1.0001rootsTemp[[2]]]<0,roots[[2]]=rootsTemp[[2]],roots[[2]]=rootsTemp[[3]]],
If[vpar[1.0001rootsTemp[[2]]]>0,roots[[1]]=rootsTemp[[2]];If[vpar[1.0001rootsTemp[[3]]]<0,roots[[2]]=rootsTemp[[3]],roots[[2]]=rootsTemp[[4]]],roots[[1]]=rootsTemp[[3]];If[vpar[1.0001rootsTemp[[4]]]<0,roots[[2]]=rootsTemp[[4]],roots[[2]]=rootsTemp[[5]]]]
]
]//Quiet;


(* ::Subsection:: *)
(*Output Second Adiabatic Invariant*)


npoints=100;precision=10;
Jinvariant=Chop@GaussLegendreQuadrature[Sqrt[1-lambda B[x]]/B[x],{x,roots[[1]],roots[[2]]},npoints,precision]*(G0+I2*r^2)
Print[Jinvariant]
