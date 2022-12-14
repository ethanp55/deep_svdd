??

??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%??L>"
Ttype0:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.7.02unknown8??
?
conv2d_70/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_70/kernel
}
$conv2d_70/kernel/Read/ReadVariableOpReadVariableOpconv2d_70/kernel*&
_output_shapes
:*
dtype0
?
batch_normalization_70/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_70/gamma
?
0batch_normalization_70/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_70/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_70/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_70/beta
?
/batch_normalization_70/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_70/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_70/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_70/moving_mean
?
6batch_normalization_70/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_70/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_70/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_70/moving_variance
?
:batch_normalization_70/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_70/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_71/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_71/kernel
}
$conv2d_71/kernel/Read/ReadVariableOpReadVariableOpconv2d_71/kernel*&
_output_shapes
:*
dtype0
?
batch_normalization_71/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_71/gamma
?
0batch_normalization_71/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_71/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_71/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_71/beta
?
/batch_normalization_71/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_71/beta*
_output_shapes
:*
dtype0
?
"batch_normalization_71/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_71/moving_mean
?
6batch_normalization_71/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_71/moving_mean*
_output_shapes
:*
dtype0
?
&batch_normalization_71/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_71/moving_variance
?
:batch_normalization_71/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_71/moving_variance*
_output_shapes
:*
dtype0
z
dense_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d@* 
shared_namedense_35/kernel
s
#dense_35/kernel/Read/ReadVariableOpReadVariableOpdense_35/kernel*
_output_shapes

:d@*
dtype0

NoOpNoOp
?(
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?'
value?'B?' B?'
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
	variables
trainable_variables
regularization_losses
	keras_api

signatures
^

kernel
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
?
axis
	gamma
beta
moving_mean
moving_variance
	variables
trainable_variables
 regularization_losses
!	keras_api
R
"	variables
#trainable_variables
$regularization_losses
%	keras_api
^

&kernel
'	variables
(trainable_variables
)regularization_losses
*	keras_api
R
+	variables
,trainable_variables
-regularization_losses
.	keras_api
?
/axis
	0gamma
1beta
2moving_mean
3moving_variance
4	variables
5trainable_variables
6regularization_losses
7	keras_api
R
8	variables
9trainable_variables
:regularization_losses
;	keras_api
R
<	variables
=trainable_variables
>regularization_losses
?	keras_api
^

@kernel
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
N
0
1
2
3
4
&5
06
17
28
39
@10

0
&1
@2
 
?
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
trainable_variables
regularization_losses
 
\Z
VARIABLE_VALUEconv2d_70/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
?
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
?
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_70/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_70/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_70/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_70/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3
 
 
?
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
	variables
trainable_variables
 regularization_losses
 
 
 
?
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
"	variables
#trainable_variables
$regularization_losses
\Z
VARIABLE_VALUEconv2d_71/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE

&0

&0
 
?
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
'	variables
(trainable_variables
)regularization_losses
 
 
 
?
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
+	variables
,trainable_variables
-regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_71/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_71/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_71/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_71/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

00
11
22
33
 
 
?
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
4	variables
5trainable_variables
6regularization_losses
 
 
 
?
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
8	variables
9trainable_variables
:regularization_losses
 
 
 
?
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
<	variables
=trainable_variables
>regularization_losses
[Y
VARIABLE_VALUEdense_35/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE

@0

@0
 
?
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
8
0
1
2
3
04
15
26
37
F
0
1
2
3
4
5
6
7
	8

9
 
 
 
 
 
 
 
 
 
 
 
 
 

0
1
2
3
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

00
11
22
33
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
?
serving_default_conv2d_70_inputPlaceholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_70_inputconv2d_70/kernelbatch_normalization_70/gammabatch_normalization_70/beta"batch_normalization_70/moving_mean&batch_normalization_70/moving_varianceconv2d_71/kernelbatch_normalization_71/gammabatch_normalization_71/beta"batch_normalization_71/moving_mean&batch_normalization_71/moving_variancedense_35/kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*-
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? */
f*R(
&__inference_signature_wrapper_37599981
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_70/kernel/Read/ReadVariableOp0batch_normalization_70/gamma/Read/ReadVariableOp/batch_normalization_70/beta/Read/ReadVariableOp6batch_normalization_70/moving_mean/Read/ReadVariableOp:batch_normalization_70/moving_variance/Read/ReadVariableOp$conv2d_71/kernel/Read/ReadVariableOp0batch_normalization_71/gamma/Read/ReadVariableOp/batch_normalization_71/beta/Read/ReadVariableOp6batch_normalization_71/moving_mean/Read/ReadVariableOp:batch_normalization_71/moving_variance/Read/ReadVariableOp#dense_35/kernel/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_save_37600546
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_70/kernelbatch_normalization_70/gammabatch_normalization_70/beta"batch_normalization_70/moving_mean&batch_normalization_70/moving_varianceconv2d_71/kernelbatch_normalization_71/gammabatch_normalization_71/beta"batch_normalization_71/moving_mean&batch_normalization_71/moving_variancedense_35/kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference__traced_restore_37600589??
?
d
H__inference_flatten_35_layer_call_and_return_conditional_losses_37600476

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????d   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????dX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
9__inference_batch_normalization_70_layer_call_fn_37600179

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_70_layer_call_and_return_conditional_losses_37599370?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_71_layer_call_and_return_conditional_losses_37599446

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_71_layer_call_and_return_conditional_losses_37599415

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
9__inference_batch_normalization_70_layer_call_fn_37600166

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_70_layer_call_and_return_conditional_losses_37599339?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_70_layer_call_and_return_conditional_losses_37599511

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
j
N__inference_max_pooling2d_71_layer_call_and_return_conditional_losses_37600465

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
9__inference_batch_normalization_71_layer_call_fn_37600347

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_71_layer_call_and_return_conditional_losses_37599446?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?.
?
K__inference_sequential_35_layer_call_and_return_conditional_losses_37599828

inputs,
conv2d_70_37599795:-
batch_normalization_70_37599799:-
batch_normalization_70_37599801:-
batch_normalization_70_37599803:-
batch_normalization_70_37599805:,
conv2d_71_37599809:-
batch_normalization_71_37599813:-
batch_normalization_71_37599815:-
batch_normalization_71_37599817:-
batch_normalization_71_37599819:#
dense_35_37599824:d@
identity??.batch_normalization_70/StatefulPartitionedCall?.batch_normalization_71/StatefulPartitionedCall?!conv2d_70/StatefulPartitionedCall?!conv2d_71/StatefulPartitionedCall? dense_35/StatefulPartitionedCall?
!conv2d_70/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_70_37599795*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_70_layer_call_and_return_conditional_losses_37599483?
leaky_re_lu_70/PartitionedCallPartitionedCall*conv2d_70/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_leaky_re_lu_70_layer_call_and_return_conditional_losses_37599492?
.batch_normalization_70/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_70/PartitionedCall:output:0batch_normalization_70_37599799batch_normalization_70_37599801batch_normalization_70_37599803batch_normalization_70_37599805*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_70_layer_call_and_return_conditional_losses_37599738?
 max_pooling2d_70/PartitionedCallPartitionedCall7batch_normalization_70/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_70_layer_call_and_return_conditional_losses_37599525?
!conv2d_71/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_70/PartitionedCall:output:0conv2d_71_37599809*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_71_layer_call_and_return_conditional_losses_37599534?
leaky_re_lu_71/PartitionedCallPartitionedCall*conv2d_71/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_leaky_re_lu_71_layer_call_and_return_conditional_losses_37599543?
.batch_normalization_71/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_71/PartitionedCall:output:0batch_normalization_71_37599813batch_normalization_71_37599815batch_normalization_71_37599817batch_normalization_71_37599819*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_71_layer_call_and_return_conditional_losses_37599675?
 max_pooling2d_71/PartitionedCallPartitionedCall7batch_normalization_71/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_71_layer_call_and_return_conditional_losses_37599576?
flatten_35/PartitionedCallPartitionedCall)max_pooling2d_71/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_flatten_35_layer_call_and_return_conditional_losses_37599584?
 dense_35/StatefulPartitionedCallStatefulPartitionedCall#flatten_35/PartitionedCall:output:0dense_35_37599824*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_35_layer_call_and_return_conditional_losses_37599593x
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp/^batch_normalization_70/StatefulPartitionedCall/^batch_normalization_71/StatefulPartitionedCall"^conv2d_70/StatefulPartitionedCall"^conv2d_71/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????: : : : : : : : : : : 2`
.batch_normalization_70/StatefulPartitionedCall.batch_normalization_70/StatefulPartitionedCall2`
.batch_normalization_71/StatefulPartitionedCall.batch_normalization_71/StatefulPartitionedCall2F
!conv2d_70/StatefulPartitionedCall!conv2d_70/StatefulPartitionedCall2F
!conv2d_71/StatefulPartitionedCall!conv2d_71/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_71_layer_call_fn_37600373

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_71_layer_call_and_return_conditional_losses_37599675w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_70_layer_call_and_return_conditional_losses_37600223

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
h
L__inference_leaky_re_lu_71_layer_call_and_return_conditional_losses_37600321

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????*
alpha%
?#<g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_conv2d_70_layer_call_and_return_conditional_losses_37600143

inputs8
conv2d_readvariableop_resource:
identity??Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:?????????^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_71_layer_call_and_return_conditional_losses_37600391

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
G__inference_conv2d_71_layer_call_and_return_conditional_losses_37600311

inputs8
conv2d_readvariableop_resource:
identity??Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:?????????^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_71_layer_call_and_return_conditional_losses_37599562

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
h
L__inference_leaky_re_lu_70_layer_call_and_return_conditional_losses_37599492

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????*
alpha%
?#<g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
9__inference_batch_normalization_71_layer_call_fn_37600334

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_71_layer_call_and_return_conditional_losses_37599415?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?

?
0__inference_sequential_35_layer_call_fn_37600035

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:#
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:d@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*-
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_sequential_35_layer_call_and_return_conditional_losses_37599828o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
j
N__inference_max_pooling2d_71_layer_call_and_return_conditional_losses_37599466

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_70_layer_call_and_return_conditional_losses_37600277

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?9
?

K__inference_sequential_35_layer_call_and_return_conditional_losses_37600129

inputsB
(conv2d_70_conv2d_readvariableop_resource:<
.batch_normalization_70_readvariableop_resource:>
0batch_normalization_70_readvariableop_1_resource:M
?batch_normalization_70_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_70_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_71_conv2d_readvariableop_resource:<
.batch_normalization_71_readvariableop_resource:>
0batch_normalization_71_readvariableop_1_resource:M
?batch_normalization_71_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_71_fusedbatchnormv3_readvariableop_1_resource:9
'dense_35_matmul_readvariableop_resource:d@
identity??6batch_normalization_70/FusedBatchNormV3/ReadVariableOp?8batch_normalization_70/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_70/ReadVariableOp?'batch_normalization_70/ReadVariableOp_1?6batch_normalization_71/FusedBatchNormV3/ReadVariableOp?8batch_normalization_71/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_71/ReadVariableOp?'batch_normalization_71/ReadVariableOp_1?conv2d_70/Conv2D/ReadVariableOp?conv2d_71/Conv2D/ReadVariableOp?dense_35/MatMul/ReadVariableOp?
conv2d_70/Conv2D/ReadVariableOpReadVariableOp(conv2d_70_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_70/Conv2DConv2Dinputs'conv2d_70/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
leaky_re_lu_70/LeakyRelu	LeakyReluconv2d_70/Conv2D:output:0*/
_output_shapes
:?????????*
alpha%
?#<?
%batch_normalization_70/ReadVariableOpReadVariableOp.batch_normalization_70_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_70/ReadVariableOp_1ReadVariableOp0batch_normalization_70_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_70/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_70_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_70/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_70_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_70/FusedBatchNormV3FusedBatchNormV3&leaky_re_lu_70/LeakyRelu:activations:0-batch_normalization_70/ReadVariableOp:value:0/batch_normalization_70/ReadVariableOp_1:value:0>batch_normalization_70/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_70/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
is_training( ?
max_pooling2d_70/MaxPoolMaxPool+batch_normalization_70/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
?
conv2d_71/Conv2D/ReadVariableOpReadVariableOp(conv2d_71_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_71/Conv2DConv2D!max_pooling2d_70/MaxPool:output:0'conv2d_71/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
leaky_re_lu_71/LeakyRelu	LeakyReluconv2d_71/Conv2D:output:0*/
_output_shapes
:?????????*
alpha%
?#<?
%batch_normalization_71/ReadVariableOpReadVariableOp.batch_normalization_71_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_71/ReadVariableOp_1ReadVariableOp0batch_normalization_71_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_71/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_71_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_71/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_71_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_71/FusedBatchNormV3FusedBatchNormV3&leaky_re_lu_71/LeakyRelu:activations:0-batch_normalization_71/ReadVariableOp:value:0/batch_normalization_71/ReadVariableOp_1:value:0>batch_normalization_71/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_71/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
is_training( ?
max_pooling2d_71/MaxPoolMaxPool+batch_normalization_71/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
a
flatten_35/ConstConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
flatten_35/ReshapeReshape!max_pooling2d_71/MaxPool:output:0flatten_35/Const:output:0*
T0*'
_output_shapes
:?????????d?
dense_35/MatMul/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource*
_output_shapes

:d@*
dtype0?
dense_35/MatMulMatMulflatten_35/Reshape:output:0&dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@h
IdentityIdentitydense_35/MatMul:product:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp7^batch_normalization_70/FusedBatchNormV3/ReadVariableOp9^batch_normalization_70/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_70/ReadVariableOp(^batch_normalization_70/ReadVariableOp_17^batch_normalization_71/FusedBatchNormV3/ReadVariableOp9^batch_normalization_71/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_71/ReadVariableOp(^batch_normalization_71/ReadVariableOp_1 ^conv2d_70/Conv2D/ReadVariableOp ^conv2d_71/Conv2D/ReadVariableOp^dense_35/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????: : : : : : : : : : : 2p
6batch_normalization_70/FusedBatchNormV3/ReadVariableOp6batch_normalization_70/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_70/FusedBatchNormV3/ReadVariableOp_18batch_normalization_70/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_70/ReadVariableOp%batch_normalization_70/ReadVariableOp2R
'batch_normalization_70/ReadVariableOp_1'batch_normalization_70/ReadVariableOp_12p
6batch_normalization_71/FusedBatchNormV3/ReadVariableOp6batch_normalization_71/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_71/FusedBatchNormV3/ReadVariableOp_18batch_normalization_71/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_71/ReadVariableOp%batch_normalization_71/ReadVariableOp2R
'batch_normalization_71/ReadVariableOp_1'batch_normalization_71/ReadVariableOp_12B
conv2d_70/Conv2D/ReadVariableOpconv2d_70/Conv2D/ReadVariableOp2B
conv2d_71/Conv2D/ReadVariableOpconv2d_71/Conv2D/ReadVariableOp2@
dense_35/MatMul/ReadVariableOpdense_35/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
j
N__inference_max_pooling2d_70_layer_call_and_return_conditional_losses_37600297

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
O
3__inference_max_pooling2d_71_layer_call_fn_37600455

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_71_layer_call_and_return_conditional_losses_37599576h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
O
3__inference_max_pooling2d_71_layer_call_fn_37600450

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_71_layer_call_and_return_conditional_losses_37599466?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?E
?
#__inference__wrapped_model_37599317
conv2d_70_inputP
6sequential_35_conv2d_70_conv2d_readvariableop_resource:J
<sequential_35_batch_normalization_70_readvariableop_resource:L
>sequential_35_batch_normalization_70_readvariableop_1_resource:[
Msequential_35_batch_normalization_70_fusedbatchnormv3_readvariableop_resource:]
Osequential_35_batch_normalization_70_fusedbatchnormv3_readvariableop_1_resource:P
6sequential_35_conv2d_71_conv2d_readvariableop_resource:J
<sequential_35_batch_normalization_71_readvariableop_resource:L
>sequential_35_batch_normalization_71_readvariableop_1_resource:[
Msequential_35_batch_normalization_71_fusedbatchnormv3_readvariableop_resource:]
Osequential_35_batch_normalization_71_fusedbatchnormv3_readvariableop_1_resource:G
5sequential_35_dense_35_matmul_readvariableop_resource:d@
identity??Dsequential_35/batch_normalization_70/FusedBatchNormV3/ReadVariableOp?Fsequential_35/batch_normalization_70/FusedBatchNormV3/ReadVariableOp_1?3sequential_35/batch_normalization_70/ReadVariableOp?5sequential_35/batch_normalization_70/ReadVariableOp_1?Dsequential_35/batch_normalization_71/FusedBatchNormV3/ReadVariableOp?Fsequential_35/batch_normalization_71/FusedBatchNormV3/ReadVariableOp_1?3sequential_35/batch_normalization_71/ReadVariableOp?5sequential_35/batch_normalization_71/ReadVariableOp_1?-sequential_35/conv2d_70/Conv2D/ReadVariableOp?-sequential_35/conv2d_71/Conv2D/ReadVariableOp?,sequential_35/dense_35/MatMul/ReadVariableOp?
-sequential_35/conv2d_70/Conv2D/ReadVariableOpReadVariableOp6sequential_35_conv2d_70_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential_35/conv2d_70/Conv2DConv2Dconv2d_70_input5sequential_35/conv2d_70/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
&sequential_35/leaky_re_lu_70/LeakyRelu	LeakyRelu'sequential_35/conv2d_70/Conv2D:output:0*/
_output_shapes
:?????????*
alpha%
?#<?
3sequential_35/batch_normalization_70/ReadVariableOpReadVariableOp<sequential_35_batch_normalization_70_readvariableop_resource*
_output_shapes
:*
dtype0?
5sequential_35/batch_normalization_70/ReadVariableOp_1ReadVariableOp>sequential_35_batch_normalization_70_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Dsequential_35/batch_normalization_70/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_35_batch_normalization_70_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Fsequential_35/batch_normalization_70/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_35_batch_normalization_70_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
5sequential_35/batch_normalization_70/FusedBatchNormV3FusedBatchNormV34sequential_35/leaky_re_lu_70/LeakyRelu:activations:0;sequential_35/batch_normalization_70/ReadVariableOp:value:0=sequential_35/batch_normalization_70/ReadVariableOp_1:value:0Lsequential_35/batch_normalization_70/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_35/batch_normalization_70/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
is_training( ?
&sequential_35/max_pooling2d_70/MaxPoolMaxPool9sequential_35/batch_normalization_70/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
?
-sequential_35/conv2d_71/Conv2D/ReadVariableOpReadVariableOp6sequential_35_conv2d_71_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential_35/conv2d_71/Conv2DConv2D/sequential_35/max_pooling2d_70/MaxPool:output:05sequential_35/conv2d_71/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
&sequential_35/leaky_re_lu_71/LeakyRelu	LeakyRelu'sequential_35/conv2d_71/Conv2D:output:0*/
_output_shapes
:?????????*
alpha%
?#<?
3sequential_35/batch_normalization_71/ReadVariableOpReadVariableOp<sequential_35_batch_normalization_71_readvariableop_resource*
_output_shapes
:*
dtype0?
5sequential_35/batch_normalization_71/ReadVariableOp_1ReadVariableOp>sequential_35_batch_normalization_71_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Dsequential_35/batch_normalization_71/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_35_batch_normalization_71_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Fsequential_35/batch_normalization_71/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_35_batch_normalization_71_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
5sequential_35/batch_normalization_71/FusedBatchNormV3FusedBatchNormV34sequential_35/leaky_re_lu_71/LeakyRelu:activations:0;sequential_35/batch_normalization_71/ReadVariableOp:value:0=sequential_35/batch_normalization_71/ReadVariableOp_1:value:0Lsequential_35/batch_normalization_71/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_35/batch_normalization_71/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
is_training( ?
&sequential_35/max_pooling2d_71/MaxPoolMaxPool9sequential_35/batch_normalization_71/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
o
sequential_35/flatten_35/ConstConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
 sequential_35/flatten_35/ReshapeReshape/sequential_35/max_pooling2d_71/MaxPool:output:0'sequential_35/flatten_35/Const:output:0*
T0*'
_output_shapes
:?????????d?
,sequential_35/dense_35/MatMul/ReadVariableOpReadVariableOp5sequential_35_dense_35_matmul_readvariableop_resource*
_output_shapes

:d@*
dtype0?
sequential_35/dense_35/MatMulMatMul)sequential_35/flatten_35/Reshape:output:04sequential_35/dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@v
IdentityIdentity'sequential_35/dense_35/MatMul:product:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOpE^sequential_35/batch_normalization_70/FusedBatchNormV3/ReadVariableOpG^sequential_35/batch_normalization_70/FusedBatchNormV3/ReadVariableOp_14^sequential_35/batch_normalization_70/ReadVariableOp6^sequential_35/batch_normalization_70/ReadVariableOp_1E^sequential_35/batch_normalization_71/FusedBatchNormV3/ReadVariableOpG^sequential_35/batch_normalization_71/FusedBatchNormV3/ReadVariableOp_14^sequential_35/batch_normalization_71/ReadVariableOp6^sequential_35/batch_normalization_71/ReadVariableOp_1.^sequential_35/conv2d_70/Conv2D/ReadVariableOp.^sequential_35/conv2d_71/Conv2D/ReadVariableOp-^sequential_35/dense_35/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????: : : : : : : : : : : 2?
Dsequential_35/batch_normalization_70/FusedBatchNormV3/ReadVariableOpDsequential_35/batch_normalization_70/FusedBatchNormV3/ReadVariableOp2?
Fsequential_35/batch_normalization_70/FusedBatchNormV3/ReadVariableOp_1Fsequential_35/batch_normalization_70/FusedBatchNormV3/ReadVariableOp_12j
3sequential_35/batch_normalization_70/ReadVariableOp3sequential_35/batch_normalization_70/ReadVariableOp2n
5sequential_35/batch_normalization_70/ReadVariableOp_15sequential_35/batch_normalization_70/ReadVariableOp_12?
Dsequential_35/batch_normalization_71/FusedBatchNormV3/ReadVariableOpDsequential_35/batch_normalization_71/FusedBatchNormV3/ReadVariableOp2?
Fsequential_35/batch_normalization_71/FusedBatchNormV3/ReadVariableOp_1Fsequential_35/batch_normalization_71/FusedBatchNormV3/ReadVariableOp_12j
3sequential_35/batch_normalization_71/ReadVariableOp3sequential_35/batch_normalization_71/ReadVariableOp2n
5sequential_35/batch_normalization_71/ReadVariableOp_15sequential_35/batch_normalization_71/ReadVariableOp_12^
-sequential_35/conv2d_70/Conv2D/ReadVariableOp-sequential_35/conv2d_70/Conv2D/ReadVariableOp2^
-sequential_35/conv2d_71/Conv2D/ReadVariableOp-sequential_35/conv2d_71/Conv2D/ReadVariableOp2\
,sequential_35/dense_35/MatMul/ReadVariableOp,sequential_35/dense_35/MatMul/ReadVariableOp:` \
/
_output_shapes
:?????????
)
_user_specified_nameconv2d_70_input
?
?
T__inference_batch_normalization_70_layer_call_and_return_conditional_losses_37600241

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_71_layer_call_and_return_conditional_losses_37600427

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
h
L__inference_leaky_re_lu_71_layer_call_and_return_conditional_losses_37599543

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????*
alpha%
?#<g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
9__inference_batch_normalization_71_layer_call_fn_37600360

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_71_layer_call_and_return_conditional_losses_37599562w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
I
-__inference_flatten_35_layer_call_fn_37600470

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_flatten_35_layer_call_and_return_conditional_losses_37599584`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_conv2d_70_layer_call_and_return_conditional_losses_37599483

inputs8
conv2d_readvariableop_resource:
identity??Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:?????????^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
h
L__inference_leaky_re_lu_70_layer_call_and_return_conditional_losses_37600153

inputs
identity_
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:?????????*
alpha%
?#<g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
&__inference_signature_wrapper_37599981
conv2d_70_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:#
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:d@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_70_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*-
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__wrapped_model_37599317o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????
)
_user_specified_nameconv2d_70_input
?
?
T__inference_batch_normalization_70_layer_call_and_return_conditional_losses_37600259

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

+__inference_dense_35_layer_call_fn_37600483

inputs
unknown:d@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_35_layer_call_and_return_conditional_losses_37599593o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????d: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
F__inference_dense_35_layer_call_and_return_conditional_losses_37599593

inputs0
matmul_readvariableop_resource:d@
identity??MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@_
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:?????????@^
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????d: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_70_layer_call_and_return_conditional_losses_37599738

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
d
H__inference_flatten_35_layer_call_and_return_conditional_losses_37599584

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????d   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????dX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_conv2d_70_layer_call_fn_37600136

inputs!
unknown:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_70_layer_call_and_return_conditional_losses_37599483w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_71_layer_call_and_return_conditional_losses_37599675

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
O
3__inference_max_pooling2d_70_layer_call_fn_37600282

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_70_layer_call_and_return_conditional_losses_37599390?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
M
1__inference_leaky_re_lu_70_layer_call_fn_37600148

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_leaky_re_lu_70_layer_call_and_return_conditional_losses_37599492h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?/
?
K__inference_sequential_35_layer_call_and_return_conditional_losses_37599952
conv2d_70_input,
conv2d_70_37599919:-
batch_normalization_70_37599923:-
batch_normalization_70_37599925:-
batch_normalization_70_37599927:-
batch_normalization_70_37599929:,
conv2d_71_37599933:-
batch_normalization_71_37599937:-
batch_normalization_71_37599939:-
batch_normalization_71_37599941:-
batch_normalization_71_37599943:#
dense_35_37599948:d@
identity??.batch_normalization_70/StatefulPartitionedCall?.batch_normalization_71/StatefulPartitionedCall?!conv2d_70/StatefulPartitionedCall?!conv2d_71/StatefulPartitionedCall? dense_35/StatefulPartitionedCall?
!conv2d_70/StatefulPartitionedCallStatefulPartitionedCallconv2d_70_inputconv2d_70_37599919*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_70_layer_call_and_return_conditional_losses_37599483?
leaky_re_lu_70/PartitionedCallPartitionedCall*conv2d_70/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_leaky_re_lu_70_layer_call_and_return_conditional_losses_37599492?
.batch_normalization_70/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_70/PartitionedCall:output:0batch_normalization_70_37599923batch_normalization_70_37599925batch_normalization_70_37599927batch_normalization_70_37599929*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_70_layer_call_and_return_conditional_losses_37599738?
 max_pooling2d_70/PartitionedCallPartitionedCall7batch_normalization_70/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_70_layer_call_and_return_conditional_losses_37599525?
!conv2d_71/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_70/PartitionedCall:output:0conv2d_71_37599933*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_71_layer_call_and_return_conditional_losses_37599534?
leaky_re_lu_71/PartitionedCallPartitionedCall*conv2d_71/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_leaky_re_lu_71_layer_call_and_return_conditional_losses_37599543?
.batch_normalization_71/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_71/PartitionedCall:output:0batch_normalization_71_37599937batch_normalization_71_37599939batch_normalization_71_37599941batch_normalization_71_37599943*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_71_layer_call_and_return_conditional_losses_37599675?
 max_pooling2d_71/PartitionedCallPartitionedCall7batch_normalization_71/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_71_layer_call_and_return_conditional_losses_37599576?
flatten_35/PartitionedCallPartitionedCall)max_pooling2d_71/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_flatten_35_layer_call_and_return_conditional_losses_37599584?
 dense_35/StatefulPartitionedCallStatefulPartitionedCall#flatten_35/PartitionedCall:output:0dense_35_37599948*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_35_layer_call_and_return_conditional_losses_37599593x
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp/^batch_normalization_70/StatefulPartitionedCall/^batch_normalization_71/StatefulPartitionedCall"^conv2d_70/StatefulPartitionedCall"^conv2d_71/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????: : : : : : : : : : : 2`
.batch_normalization_70/StatefulPartitionedCall.batch_normalization_70/StatefulPartitionedCall2`
.batch_normalization_71/StatefulPartitionedCall.batch_normalization_71/StatefulPartitionedCall2F
!conv2d_70/StatefulPartitionedCall!conv2d_70/StatefulPartitionedCall2F
!conv2d_71/StatefulPartitionedCall!conv2d_71/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall:` \
/
_output_shapes
:?????????
)
_user_specified_nameconv2d_70_input
?
j
N__inference_max_pooling2d_70_layer_call_and_return_conditional_losses_37599525

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
0__inference_sequential_35_layer_call_fn_37599623
conv2d_70_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:#
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:d@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_70_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*-
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_sequential_35_layer_call_and_return_conditional_losses_37599598o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????
)
_user_specified_nameconv2d_70_input
?
?
9__inference_batch_normalization_70_layer_call_fn_37600192

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_70_layer_call_and_return_conditional_losses_37599511w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_70_layer_call_and_return_conditional_losses_37599339

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
j
N__inference_max_pooling2d_70_layer_call_and_return_conditional_losses_37600292

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
0__inference_sequential_35_layer_call_fn_37599880
conv2d_70_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:#
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:d@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_70_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*-
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_sequential_35_layer_call_and_return_conditional_losses_37599828o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????
)
_user_specified_nameconv2d_70_input
?
?
G__inference_conv2d_71_layer_call_and_return_conditional_losses_37599534

inputs8
conv2d_readvariableop_resource:
identity??Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
f
IdentityIdentityConv2D:output:0^NoOp*
T0*/
_output_shapes
:?????????^
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
j
N__inference_max_pooling2d_71_layer_call_and_return_conditional_losses_37600460

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_71_layer_call_and_return_conditional_losses_37600409

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
j
N__inference_max_pooling2d_71_layer_call_and_return_conditional_losses_37599576

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
j
N__inference_max_pooling2d_70_layer_call_and_return_conditional_losses_37599390

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?1
?
$__inference__traced_restore_37600589
file_prefix;
!assignvariableop_conv2d_70_kernel:=
/assignvariableop_1_batch_normalization_70_gamma:<
.assignvariableop_2_batch_normalization_70_beta:C
5assignvariableop_3_batch_normalization_70_moving_mean:G
9assignvariableop_4_batch_normalization_70_moving_variance:=
#assignvariableop_5_conv2d_71_kernel:=
/assignvariableop_6_batch_normalization_71_gamma:<
.assignvariableop_7_batch_normalization_71_beta:C
5assignvariableop_8_batch_normalization_71_moving_mean:G
9assignvariableop_9_batch_normalization_71_moving_variance:5
#assignvariableop_10_dense_35_kernel:d@
identity_12??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*+
value"B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*D
_output_shapes2
0::::::::::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_70_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp/assignvariableop_1_batch_normalization_70_gammaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp.assignvariableop_2_batch_normalization_70_betaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp5assignvariableop_3_batch_normalization_70_moving_meanIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp9assignvariableop_4_batch_normalization_70_moving_varianceIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp#assignvariableop_5_conv2d_71_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp/assignvariableop_6_batch_normalization_71_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp.assignvariableop_7_batch_normalization_71_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp5assignvariableop_8_batch_normalization_71_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp9assignvariableop_9_batch_normalization_71_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_35_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_11Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_12IdentityIdentity_11:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_12Identity_12:output:0*+
_input_shapes
: : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
O
3__inference_max_pooling2d_70_layer_call_fn_37600287

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_70_layer_call_and_return_conditional_losses_37599525h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
M
1__inference_leaky_re_lu_71_layer_call_fn_37600316

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_leaky_re_lu_71_layer_call_and_return_conditional_losses_37599543h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?9
?

K__inference_sequential_35_layer_call_and_return_conditional_losses_37600082

inputsB
(conv2d_70_conv2d_readvariableop_resource:<
.batch_normalization_70_readvariableop_resource:>
0batch_normalization_70_readvariableop_1_resource:M
?batch_normalization_70_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_70_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_71_conv2d_readvariableop_resource:<
.batch_normalization_71_readvariableop_resource:>
0batch_normalization_71_readvariableop_1_resource:M
?batch_normalization_71_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_71_fusedbatchnormv3_readvariableop_1_resource:9
'dense_35_matmul_readvariableop_resource:d@
identity??6batch_normalization_70/FusedBatchNormV3/ReadVariableOp?8batch_normalization_70/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_70/ReadVariableOp?'batch_normalization_70/ReadVariableOp_1?6batch_normalization_71/FusedBatchNormV3/ReadVariableOp?8batch_normalization_71/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_71/ReadVariableOp?'batch_normalization_71/ReadVariableOp_1?conv2d_70/Conv2D/ReadVariableOp?conv2d_71/Conv2D/ReadVariableOp?dense_35/MatMul/ReadVariableOp?
conv2d_70/Conv2D/ReadVariableOpReadVariableOp(conv2d_70_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_70/Conv2DConv2Dinputs'conv2d_70/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
leaky_re_lu_70/LeakyRelu	LeakyReluconv2d_70/Conv2D:output:0*/
_output_shapes
:?????????*
alpha%
?#<?
%batch_normalization_70/ReadVariableOpReadVariableOp.batch_normalization_70_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_70/ReadVariableOp_1ReadVariableOp0batch_normalization_70_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_70/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_70_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_70/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_70_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_70/FusedBatchNormV3FusedBatchNormV3&leaky_re_lu_70/LeakyRelu:activations:0-batch_normalization_70/ReadVariableOp:value:0/batch_normalization_70/ReadVariableOp_1:value:0>batch_normalization_70/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_70/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
is_training( ?
max_pooling2d_70/MaxPoolMaxPool+batch_normalization_70/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
?
conv2d_71/Conv2D/ReadVariableOpReadVariableOp(conv2d_71_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_71/Conv2DConv2D!max_pooling2d_70/MaxPool:output:0'conv2d_71/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
leaky_re_lu_71/LeakyRelu	LeakyReluconv2d_71/Conv2D:output:0*/
_output_shapes
:?????????*
alpha%
?#<?
%batch_normalization_71/ReadVariableOpReadVariableOp.batch_normalization_71_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_71/ReadVariableOp_1ReadVariableOp0batch_normalization_71_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_71/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_71_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_71/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_71_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_71/FusedBatchNormV3FusedBatchNormV3&leaky_re_lu_71/LeakyRelu:activations:0-batch_normalization_71/ReadVariableOp:value:0/batch_normalization_71/ReadVariableOp_1:value:0>batch_normalization_71/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_71/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
is_training( ?
max_pooling2d_71/MaxPoolMaxPool+batch_normalization_71/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
a
flatten_35/ConstConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
flatten_35/ReshapeReshape!max_pooling2d_71/MaxPool:output:0flatten_35/Const:output:0*
T0*'
_output_shapes
:?????????d?
dense_35/MatMul/ReadVariableOpReadVariableOp'dense_35_matmul_readvariableop_resource*
_output_shapes

:d@*
dtype0?
dense_35/MatMulMatMulflatten_35/Reshape:output:0&dense_35/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@h
IdentityIdentitydense_35/MatMul:product:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp7^batch_normalization_70/FusedBatchNormV3/ReadVariableOp9^batch_normalization_70/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_70/ReadVariableOp(^batch_normalization_70/ReadVariableOp_17^batch_normalization_71/FusedBatchNormV3/ReadVariableOp9^batch_normalization_71/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_71/ReadVariableOp(^batch_normalization_71/ReadVariableOp_1 ^conv2d_70/Conv2D/ReadVariableOp ^conv2d_71/Conv2D/ReadVariableOp^dense_35/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????: : : : : : : : : : : 2p
6batch_normalization_70/FusedBatchNormV3/ReadVariableOp6batch_normalization_70/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_70/FusedBatchNormV3/ReadVariableOp_18batch_normalization_70/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_70/ReadVariableOp%batch_normalization_70/ReadVariableOp2R
'batch_normalization_70/ReadVariableOp_1'batch_normalization_70/ReadVariableOp_12p
6batch_normalization_71/FusedBatchNormV3/ReadVariableOp6batch_normalization_71/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_71/FusedBatchNormV3/ReadVariableOp_18batch_normalization_71/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_71/ReadVariableOp%batch_normalization_71/ReadVariableOp2R
'batch_normalization_71/ReadVariableOp_1'batch_normalization_71/ReadVariableOp_12B
conv2d_70/Conv2D/ReadVariableOpconv2d_70/Conv2D/ReadVariableOp2B
conv2d_71/Conv2D/ReadVariableOpconv2d_71/Conv2D/ReadVariableOp2@
dense_35/MatMul/ReadVariableOpdense_35/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_conv2d_71_layer_call_fn_37600304

inputs!
unknown:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_71_layer_call_and_return_conditional_losses_37599534w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
F__inference_dense_35_layer_call_and_return_conditional_losses_37600490

inputs0
matmul_readvariableop_resource:d@
identity??MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@_
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:?????????@^
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????d: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?$
?
!__inference__traced_save_37600546
file_prefix/
+savev2_conv2d_70_kernel_read_readvariableop;
7savev2_batch_normalization_70_gamma_read_readvariableop:
6savev2_batch_normalization_70_beta_read_readvariableopA
=savev2_batch_normalization_70_moving_mean_read_readvariableopE
Asavev2_batch_normalization_70_moving_variance_read_readvariableop/
+savev2_conv2d_71_kernel_read_readvariableop;
7savev2_batch_normalization_71_gamma_read_readvariableop:
6savev2_batch_normalization_71_beta_read_readvariableopA
=savev2_batch_normalization_71_moving_mean_read_readvariableopE
Asavev2_batch_normalization_71_moving_variance_read_readvariableop.
*savev2_dense_35_kernel_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*+
value"B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_70_kernel_read_readvariableop7savev2_batch_normalization_70_gamma_read_readvariableop6savev2_batch_normalization_70_beta_read_readvariableop=savev2_batch_normalization_70_moving_mean_read_readvariableopAsavev2_batch_normalization_70_moving_variance_read_readvariableop+savev2_conv2d_71_kernel_read_readvariableop7savev2_batch_normalization_71_gamma_read_readvariableop6savev2_batch_normalization_71_beta_read_readvariableop=savev2_batch_normalization_71_moving_mean_read_readvariableopAsavev2_batch_normalization_71_moving_variance_read_readvariableop*savev2_dense_35_kernel_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*u
_input_shapesd
b: :::::::::::d@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 	

_output_shapes
:: 


_output_shapes
::$ 

_output_shapes

:d@:

_output_shapes
: 
?.
?
K__inference_sequential_35_layer_call_and_return_conditional_losses_37599598

inputs,
conv2d_70_37599484:-
batch_normalization_70_37599512:-
batch_normalization_70_37599514:-
batch_normalization_70_37599516:-
batch_normalization_70_37599518:,
conv2d_71_37599535:-
batch_normalization_71_37599563:-
batch_normalization_71_37599565:-
batch_normalization_71_37599567:-
batch_normalization_71_37599569:#
dense_35_37599594:d@
identity??.batch_normalization_70/StatefulPartitionedCall?.batch_normalization_71/StatefulPartitionedCall?!conv2d_70/StatefulPartitionedCall?!conv2d_71/StatefulPartitionedCall? dense_35/StatefulPartitionedCall?
!conv2d_70/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_70_37599484*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_70_layer_call_and_return_conditional_losses_37599483?
leaky_re_lu_70/PartitionedCallPartitionedCall*conv2d_70/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_leaky_re_lu_70_layer_call_and_return_conditional_losses_37599492?
.batch_normalization_70/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_70/PartitionedCall:output:0batch_normalization_70_37599512batch_normalization_70_37599514batch_normalization_70_37599516batch_normalization_70_37599518*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_70_layer_call_and_return_conditional_losses_37599511?
 max_pooling2d_70/PartitionedCallPartitionedCall7batch_normalization_70/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_70_layer_call_and_return_conditional_losses_37599525?
!conv2d_71/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_70/PartitionedCall:output:0conv2d_71_37599535*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_71_layer_call_and_return_conditional_losses_37599534?
leaky_re_lu_71/PartitionedCallPartitionedCall*conv2d_71/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_leaky_re_lu_71_layer_call_and_return_conditional_losses_37599543?
.batch_normalization_71/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_71/PartitionedCall:output:0batch_normalization_71_37599563batch_normalization_71_37599565batch_normalization_71_37599567batch_normalization_71_37599569*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_71_layer_call_and_return_conditional_losses_37599562?
 max_pooling2d_71/PartitionedCallPartitionedCall7batch_normalization_71/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_71_layer_call_and_return_conditional_losses_37599576?
flatten_35/PartitionedCallPartitionedCall)max_pooling2d_71/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_flatten_35_layer_call_and_return_conditional_losses_37599584?
 dense_35/StatefulPartitionedCallStatefulPartitionedCall#flatten_35/PartitionedCall:output:0dense_35_37599594*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_35_layer_call_and_return_conditional_losses_37599593x
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp/^batch_normalization_70/StatefulPartitionedCall/^batch_normalization_71/StatefulPartitionedCall"^conv2d_70/StatefulPartitionedCall"^conv2d_71/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????: : : : : : : : : : : 2`
.batch_normalization_70/StatefulPartitionedCall.batch_normalization_70/StatefulPartitionedCall2`
.batch_normalization_71/StatefulPartitionedCall.batch_normalization_71/StatefulPartitionedCall2F
!conv2d_70/StatefulPartitionedCall!conv2d_70/StatefulPartitionedCall2F
!conv2d_71/StatefulPartitionedCall!conv2d_71/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
0__inference_sequential_35_layer_call_fn_37600008

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:#
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:d@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*-
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_sequential_35_layer_call_and_return_conditional_losses_37599598o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????: : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_71_layer_call_and_return_conditional_losses_37600445

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:??????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
T__inference_batch_normalization_70_layer_call_and_return_conditional_losses_37599370

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?/
?
K__inference_sequential_35_layer_call_and_return_conditional_losses_37599916
conv2d_70_input,
conv2d_70_37599883:-
batch_normalization_70_37599887:-
batch_normalization_70_37599889:-
batch_normalization_70_37599891:-
batch_normalization_70_37599893:,
conv2d_71_37599897:-
batch_normalization_71_37599901:-
batch_normalization_71_37599903:-
batch_normalization_71_37599905:-
batch_normalization_71_37599907:#
dense_35_37599912:d@
identity??.batch_normalization_70/StatefulPartitionedCall?.batch_normalization_71/StatefulPartitionedCall?!conv2d_70/StatefulPartitionedCall?!conv2d_71/StatefulPartitionedCall? dense_35/StatefulPartitionedCall?
!conv2d_70/StatefulPartitionedCallStatefulPartitionedCallconv2d_70_inputconv2d_70_37599883*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_70_layer_call_and_return_conditional_losses_37599483?
leaky_re_lu_70/PartitionedCallPartitionedCall*conv2d_70/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_leaky_re_lu_70_layer_call_and_return_conditional_losses_37599492?
.batch_normalization_70/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_70/PartitionedCall:output:0batch_normalization_70_37599887batch_normalization_70_37599889batch_normalization_70_37599891batch_normalization_70_37599893*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_70_layer_call_and_return_conditional_losses_37599511?
 max_pooling2d_70/PartitionedCallPartitionedCall7batch_normalization_70/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_70_layer_call_and_return_conditional_losses_37599525?
!conv2d_71/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_70/PartitionedCall:output:0conv2d_71_37599897*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_71_layer_call_and_return_conditional_losses_37599534?
leaky_re_lu_71/PartitionedCallPartitionedCall*conv2d_71/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_leaky_re_lu_71_layer_call_and_return_conditional_losses_37599543?
.batch_normalization_71/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_71/PartitionedCall:output:0batch_normalization_71_37599901batch_normalization_71_37599903batch_normalization_71_37599905batch_normalization_71_37599907*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_71_layer_call_and_return_conditional_losses_37599562?
 max_pooling2d_71/PartitionedCallPartitionedCall7batch_normalization_71/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_71_layer_call_and_return_conditional_losses_37599576?
flatten_35/PartitionedCallPartitionedCall)max_pooling2d_71/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????d* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_flatten_35_layer_call_and_return_conditional_losses_37599584?
 dense_35/StatefulPartitionedCallStatefulPartitionedCall#flatten_35/PartitionedCall:output:0dense_35_37599912*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_35_layer_call_and_return_conditional_losses_37599593x
IdentityIdentity)dense_35/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp/^batch_normalization_70/StatefulPartitionedCall/^batch_normalization_71/StatefulPartitionedCall"^conv2d_70/StatefulPartitionedCall"^conv2d_71/StatefulPartitionedCall!^dense_35/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????: : : : : : : : : : : 2`
.batch_normalization_70/StatefulPartitionedCall.batch_normalization_70/StatefulPartitionedCall2`
.batch_normalization_71/StatefulPartitionedCall.batch_normalization_71/StatefulPartitionedCall2F
!conv2d_70/StatefulPartitionedCall!conv2d_70/StatefulPartitionedCall2F
!conv2d_71/StatefulPartitionedCall!conv2d_71/StatefulPartitionedCall2D
 dense_35/StatefulPartitionedCall dense_35/StatefulPartitionedCall:` \
/
_output_shapes
:?????????
)
_user_specified_nameconv2d_70_input
?
?
9__inference_batch_normalization_70_layer_call_fn_37600205

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *]
fXRV
T__inference_batch_normalization_70_layer_call_and_return_conditional_losses_37599738w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
S
conv2d_70_input@
!serving_default_conv2d_70_input:0?????????<
dense_350
StatefulPartitionedCall:0?????????@tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
	variables
trainable_variables
regularization_losses
	keras_api

signatures
|__call__
*}&call_and_return_all_conditional_losses
~_default_save_signature"
_tf_keras_sequential
?

kernel
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
axis
	gamma
beta
moving_mean
moving_variance
	variables
trainable_variables
 regularization_losses
!	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
"	variables
#trainable_variables
$regularization_losses
%	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

&kernel
'	variables
(trainable_variables
)regularization_losses
*	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
+	variables
,trainable_variables
-regularization_losses
.	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
/axis
	0gamma
1beta
2moving_mean
3moving_variance
4	variables
5trainable_variables
6regularization_losses
7	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
8	variables
9trainable_variables
:regularization_losses
;	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
<	variables
=trainable_variables
>regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

@kernel
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
n
0
1
2
3
4
&5
06
17
28
39
@10"
trackable_list_wrapper
5
0
&1
@2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
trainable_variables
regularization_losses
|__call__
~_default_save_signature
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
*:(2conv2d_70/kernel
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_70/gamma
):'2batch_normalization_70/beta
2:0 (2"batch_normalization_70/moving_mean
6:4 (2&batch_normalization_70/moving_variance
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
	variables
trainable_variables
 regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
"	variables
#trainable_variables
$regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_71/kernel
'
&0"
trackable_list_wrapper
'
&0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
'	variables
(trainable_variables
)regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
+	variables
,trainable_variables
-regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_71/gamma
):'2batch_normalization_71/beta
2:0 (2"batch_normalization_71/moving_mean
6:4 (2&batch_normalization_71/moving_variance
<
00
11
22
33"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
4	variables
5trainable_variables
6regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
8	variables
9trainable_variables
:regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
rnon_trainable_variables

slayers
tmetrics
ulayer_regularization_losses
vlayer_metrics
<	variables
=trainable_variables
>regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:d@2dense_35/kernel
'
@0"
trackable_list_wrapper
'
@0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
wnon_trainable_variables

xlayers
ymetrics
zlayer_regularization_losses
{layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
X
0
1
2
3
04
15
26
37"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
00
11
22
33"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?2?
0__inference_sequential_35_layer_call_fn_37599623
0__inference_sequential_35_layer_call_fn_37600008
0__inference_sequential_35_layer_call_fn_37600035
0__inference_sequential_35_layer_call_fn_37599880?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
K__inference_sequential_35_layer_call_and_return_conditional_losses_37600082
K__inference_sequential_35_layer_call_and_return_conditional_losses_37600129
K__inference_sequential_35_layer_call_and_return_conditional_losses_37599916
K__inference_sequential_35_layer_call_and_return_conditional_losses_37599952?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
#__inference__wrapped_model_37599317conv2d_70_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_conv2d_70_layer_call_fn_37600136?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_conv2d_70_layer_call_and_return_conditional_losses_37600143?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_leaky_re_lu_70_layer_call_fn_37600148?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_leaky_re_lu_70_layer_call_and_return_conditional_losses_37600153?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
9__inference_batch_normalization_70_layer_call_fn_37600166
9__inference_batch_normalization_70_layer_call_fn_37600179
9__inference_batch_normalization_70_layer_call_fn_37600192
9__inference_batch_normalization_70_layer_call_fn_37600205?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
T__inference_batch_normalization_70_layer_call_and_return_conditional_losses_37600223
T__inference_batch_normalization_70_layer_call_and_return_conditional_losses_37600241
T__inference_batch_normalization_70_layer_call_and_return_conditional_losses_37600259
T__inference_batch_normalization_70_layer_call_and_return_conditional_losses_37600277?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
3__inference_max_pooling2d_70_layer_call_fn_37600282
3__inference_max_pooling2d_70_layer_call_fn_37600287?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
N__inference_max_pooling2d_70_layer_call_and_return_conditional_losses_37600292
N__inference_max_pooling2d_70_layer_call_and_return_conditional_losses_37600297?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_conv2d_71_layer_call_fn_37600304?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_conv2d_71_layer_call_and_return_conditional_losses_37600311?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_leaky_re_lu_71_layer_call_fn_37600316?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_leaky_re_lu_71_layer_call_and_return_conditional_losses_37600321?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
9__inference_batch_normalization_71_layer_call_fn_37600334
9__inference_batch_normalization_71_layer_call_fn_37600347
9__inference_batch_normalization_71_layer_call_fn_37600360
9__inference_batch_normalization_71_layer_call_fn_37600373?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
T__inference_batch_normalization_71_layer_call_and_return_conditional_losses_37600391
T__inference_batch_normalization_71_layer_call_and_return_conditional_losses_37600409
T__inference_batch_normalization_71_layer_call_and_return_conditional_losses_37600427
T__inference_batch_normalization_71_layer_call_and_return_conditional_losses_37600445?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
3__inference_max_pooling2d_71_layer_call_fn_37600450
3__inference_max_pooling2d_71_layer_call_fn_37600455?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
N__inference_max_pooling2d_71_layer_call_and_return_conditional_losses_37600460
N__inference_max_pooling2d_71_layer_call_and_return_conditional_losses_37600465?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_flatten_35_layer_call_fn_37600470?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_flatten_35_layer_call_and_return_conditional_losses_37600476?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_35_layer_call_fn_37600483?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_35_layer_call_and_return_conditional_losses_37600490?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_signature_wrapper_37599981conv2d_70_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
#__inference__wrapped_model_37599317?&0123@@?=
6?3
1?.
conv2d_70_input?????????
? "3?0
.
dense_35"?
dense_35?????????@?
T__inference_batch_normalization_70_layer_call_and_return_conditional_losses_37600223?M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
T__inference_batch_normalization_70_layer_call_and_return_conditional_losses_37600241?M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
T__inference_batch_normalization_70_layer_call_and_return_conditional_losses_37600259r;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
T__inference_batch_normalization_70_layer_call_and_return_conditional_losses_37600277r;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
9__inference_batch_normalization_70_layer_call_fn_37600166?M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
9__inference_batch_normalization_70_layer_call_fn_37600179?M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
9__inference_batch_normalization_70_layer_call_fn_37600192e;?8
1?.
(?%
inputs?????????
p 
? " ???????????
9__inference_batch_normalization_70_layer_call_fn_37600205e;?8
1?.
(?%
inputs?????????
p
? " ???????????
T__inference_batch_normalization_71_layer_call_and_return_conditional_losses_37600391?0123M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
T__inference_batch_normalization_71_layer_call_and_return_conditional_losses_37600409?0123M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
T__inference_batch_normalization_71_layer_call_and_return_conditional_losses_37600427r0123;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
T__inference_batch_normalization_71_layer_call_and_return_conditional_losses_37600445r0123;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
9__inference_batch_normalization_71_layer_call_fn_37600334?0123M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
9__inference_batch_normalization_71_layer_call_fn_37600347?0123M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
9__inference_batch_normalization_71_layer_call_fn_37600360e0123;?8
1?.
(?%
inputs?????????
p 
? " ???????????
9__inference_batch_normalization_71_layer_call_fn_37600373e0123;?8
1?.
(?%
inputs?????????
p
? " ???????????
G__inference_conv2d_70_layer_call_and_return_conditional_losses_37600143k7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
,__inference_conv2d_70_layer_call_fn_37600136^7?4
-?*
(?%
inputs?????????
? " ???????????
G__inference_conv2d_71_layer_call_and_return_conditional_losses_37600311k&7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
,__inference_conv2d_71_layer_call_fn_37600304^&7?4
-?*
(?%
inputs?????????
? " ???????????
F__inference_dense_35_layer_call_and_return_conditional_losses_37600490[@/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????@
? }
+__inference_dense_35_layer_call_fn_37600483N@/?,
%?"
 ?
inputs?????????d
? "??????????@?
H__inference_flatten_35_layer_call_and_return_conditional_losses_37600476`7?4
-?*
(?%
inputs?????????
? "%?"
?
0?????????d
? ?
-__inference_flatten_35_layer_call_fn_37600470S7?4
-?*
(?%
inputs?????????
? "??????????d?
L__inference_leaky_re_lu_70_layer_call_and_return_conditional_losses_37600153h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
1__inference_leaky_re_lu_70_layer_call_fn_37600148[7?4
-?*
(?%
inputs?????????
? " ???????????
L__inference_leaky_re_lu_71_layer_call_and_return_conditional_losses_37600321h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
1__inference_leaky_re_lu_71_layer_call_fn_37600316[7?4
-?*
(?%
inputs?????????
? " ???????????
N__inference_max_pooling2d_70_layer_call_and_return_conditional_losses_37600292?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
N__inference_max_pooling2d_70_layer_call_and_return_conditional_losses_37600297h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
3__inference_max_pooling2d_70_layer_call_fn_37600282?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
3__inference_max_pooling2d_70_layer_call_fn_37600287[7?4
-?*
(?%
inputs?????????
? " ???????????
N__inference_max_pooling2d_71_layer_call_and_return_conditional_losses_37600460?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
N__inference_max_pooling2d_71_layer_call_and_return_conditional_losses_37600465h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
3__inference_max_pooling2d_71_layer_call_fn_37600450?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
3__inference_max_pooling2d_71_layer_call_fn_37600455[7?4
-?*
(?%
inputs?????????
? " ???????????
K__inference_sequential_35_layer_call_and_return_conditional_losses_37599916~&0123@H?E
>?;
1?.
conv2d_70_input?????????
p 

 
? "%?"
?
0?????????@
? ?
K__inference_sequential_35_layer_call_and_return_conditional_losses_37599952~&0123@H?E
>?;
1?.
conv2d_70_input?????????
p

 
? "%?"
?
0?????????@
? ?
K__inference_sequential_35_layer_call_and_return_conditional_losses_37600082u&0123@??<
5?2
(?%
inputs?????????
p 

 
? "%?"
?
0?????????@
? ?
K__inference_sequential_35_layer_call_and_return_conditional_losses_37600129u&0123@??<
5?2
(?%
inputs?????????
p

 
? "%?"
?
0?????????@
? ?
0__inference_sequential_35_layer_call_fn_37599623q&0123@H?E
>?;
1?.
conv2d_70_input?????????
p 

 
? "??????????@?
0__inference_sequential_35_layer_call_fn_37599880q&0123@H?E
>?;
1?.
conv2d_70_input?????????
p

 
? "??????????@?
0__inference_sequential_35_layer_call_fn_37600008h&0123@??<
5?2
(?%
inputs?????????
p 

 
? "??????????@?
0__inference_sequential_35_layer_call_fn_37600035h&0123@??<
5?2
(?%
inputs?????????
p

 
? "??????????@?
&__inference_signature_wrapper_37599981?&0123@S?P
? 
I?F
D
conv2d_70_input1?.
conv2d_70_input?????????"3?0
.
dense_35"?
dense_35?????????@