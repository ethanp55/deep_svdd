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
 ?"serve*2.7.02unknown8ؾ
?
conv2d_106/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_106/kernel

%conv2d_106/kernel/Read/ReadVariableOpReadVariableOpconv2d_106/kernel*&
_output_shapes
:*
dtype0
?
batch_normalization_106/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebatch_normalization_106/gamma
?
1batch_normalization_106/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_106/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_106/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_106/beta
?
0batch_normalization_106/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_106/beta*
_output_shapes
:*
dtype0
?
#batch_normalization_106/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization_106/moving_mean
?
7batch_normalization_106/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_106/moving_mean*
_output_shapes
:*
dtype0
?
'batch_normalization_106/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'batch_normalization_106/moving_variance
?
;batch_normalization_106/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_106/moving_variance*
_output_shapes
:*
dtype0
?
conv2d_107/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_107/kernel

%conv2d_107/kernel/Read/ReadVariableOpReadVariableOpconv2d_107/kernel*&
_output_shapes
:*
dtype0
?
batch_normalization_107/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namebatch_normalization_107/gamma
?
1batch_normalization_107/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_107/gamma*
_output_shapes
:*
dtype0
?
batch_normalization_107/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_107/beta
?
0batch_normalization_107/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_107/beta*
_output_shapes
:*
dtype0
?
#batch_normalization_107/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization_107/moving_mean
?
7batch_normalization_107/moving_mean/Read/ReadVariableOpReadVariableOp#batch_normalization_107/moving_mean*
_output_shapes
:*
dtype0
?
'batch_normalization_107/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'batch_normalization_107/moving_variance
?
;batch_normalization_107/moving_variance/Read/ReadVariableOpReadVariableOp'batch_normalization_107/moving_variance*
_output_shapes
:*
dtype0
z
dense_53/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d@* 
shared_namedense_53/kernel
s
#dense_53/kernel/Read/ReadVariableOpReadVariableOpdense_53/kernel*
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
][
VARIABLE_VALUEconv2d_106/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
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
hf
VARIABLE_VALUEbatch_normalization_106/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_106/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE#batch_normalization_106/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE'batch_normalization_106/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

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
][
VARIABLE_VALUEconv2d_107/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
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
hf
VARIABLE_VALUEbatch_normalization_107/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEbatch_normalization_107/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE#batch_normalization_107/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE'batch_normalization_107/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

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
VARIABLE_VALUEdense_53/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
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
 serving_default_conv2d_106_inputPlaceholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCall serving_default_conv2d_106_inputconv2d_106/kernelbatch_normalization_106/gammabatch_normalization_106/beta#batch_normalization_106/moving_mean'batch_normalization_106/moving_varianceconv2d_107/kernelbatch_normalization_107/gammabatch_normalization_107/beta#batch_normalization_107/moving_mean'batch_normalization_107/moving_variancedense_53/kernel*
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
&__inference_signature_wrapper_56908824
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%conv2d_106/kernel/Read/ReadVariableOp1batch_normalization_106/gamma/Read/ReadVariableOp0batch_normalization_106/beta/Read/ReadVariableOp7batch_normalization_106/moving_mean/Read/ReadVariableOp;batch_normalization_106/moving_variance/Read/ReadVariableOp%conv2d_107/kernel/Read/ReadVariableOp1batch_normalization_107/gamma/Read/ReadVariableOp0batch_normalization_107/beta/Read/ReadVariableOp7batch_normalization_107/moving_mean/Read/ReadVariableOp;batch_normalization_107/moving_variance/Read/ReadVariableOp#dense_53/kernel/Read/ReadVariableOpConst*
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
!__inference__traced_save_56909389
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_106/kernelbatch_normalization_106/gammabatch_normalization_106/beta#batch_normalization_106/moving_mean'batch_normalization_106/moving_varianceconv2d_107/kernelbatch_normalization_107/gammabatch_normalization_107/beta#batch_normalization_107/moving_mean'batch_normalization_107/moving_variancedense_53/kernel*
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
$__inference__traced_restore_56909432??
?/
?
K__inference_sequential_53_layer_call_and_return_conditional_losses_56908759
conv2d_106_input-
conv2d_106_56908726:.
 batch_normalization_106_56908730:.
 batch_normalization_106_56908732:.
 batch_normalization_106_56908734:.
 batch_normalization_106_56908736:-
conv2d_107_56908740:.
 batch_normalization_107_56908744:.
 batch_normalization_107_56908746:.
 batch_normalization_107_56908748:.
 batch_normalization_107_56908750:#
dense_53_56908755:d@
identity??/batch_normalization_106/StatefulPartitionedCall?/batch_normalization_107/StatefulPartitionedCall?"conv2d_106/StatefulPartitionedCall?"conv2d_107/StatefulPartitionedCall? dense_53/StatefulPartitionedCall?
"conv2d_106/StatefulPartitionedCallStatefulPartitionedCallconv2d_106_inputconv2d_106_56908726*
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
GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_106_layer_call_and_return_conditional_losses_56908326?
leaky_re_lu_106/PartitionedCallPartitionedCall+conv2d_106/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *V
fQRO
M__inference_leaky_re_lu_106_layer_call_and_return_conditional_losses_56908335?
/batch_normalization_106/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_106/PartitionedCall:output:0 batch_normalization_106_56908730 batch_normalization_106_56908732 batch_normalization_106_56908734 batch_normalization_106_56908736*
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
GPU2*0J 8? *^
fYRW
U__inference_batch_normalization_106_layer_call_and_return_conditional_losses_56908354?
!max_pooling2d_106/PartitionedCallPartitionedCall8batch_normalization_106/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *X
fSRQ
O__inference_max_pooling2d_106_layer_call_and_return_conditional_losses_56908368?
"conv2d_107/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_106/PartitionedCall:output:0conv2d_107_56908740*
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
GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_107_layer_call_and_return_conditional_losses_56908377?
leaky_re_lu_107/PartitionedCallPartitionedCall+conv2d_107/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *V
fQRO
M__inference_leaky_re_lu_107_layer_call_and_return_conditional_losses_56908386?
/batch_normalization_107/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_107/PartitionedCall:output:0 batch_normalization_107_56908744 batch_normalization_107_56908746 batch_normalization_107_56908748 batch_normalization_107_56908750*
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
GPU2*0J 8? *^
fYRW
U__inference_batch_normalization_107_layer_call_and_return_conditional_losses_56908405?
!max_pooling2d_107/PartitionedCallPartitionedCall8batch_normalization_107/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *X
fSRQ
O__inference_max_pooling2d_107_layer_call_and_return_conditional_losses_56908419?
flatten_53/PartitionedCallPartitionedCall*max_pooling2d_107/PartitionedCall:output:0*
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
H__inference_flatten_53_layer_call_and_return_conditional_losses_56908427?
 dense_53/StatefulPartitionedCallStatefulPartitionedCall#flatten_53/PartitionedCall:output:0dense_53_56908755*
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
F__inference_dense_53_layer_call_and_return_conditional_losses_56908436x
IdentityIdentity)dense_53/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp0^batch_normalization_106/StatefulPartitionedCall0^batch_normalization_107/StatefulPartitionedCall#^conv2d_106/StatefulPartitionedCall#^conv2d_107/StatefulPartitionedCall!^dense_53/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????: : : : : : : : : : : 2b
/batch_normalization_106/StatefulPartitionedCall/batch_normalization_106/StatefulPartitionedCall2b
/batch_normalization_107/StatefulPartitionedCall/batch_normalization_107/StatefulPartitionedCall2H
"conv2d_106/StatefulPartitionedCall"conv2d_106/StatefulPartitionedCall2H
"conv2d_107/StatefulPartitionedCall"conv2d_107/StatefulPartitionedCall2D
 dense_53/StatefulPartitionedCall dense_53/StatefulPartitionedCall:a ]
/
_output_shapes
:?????????
*
_user_specified_nameconv2d_106_input
?
k
O__inference_max_pooling2d_106_layer_call_and_return_conditional_losses_56909140

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
k
O__inference_max_pooling2d_106_layer_call_and_return_conditional_losses_56908233

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
?
?
:__inference_batch_normalization_107_layer_call_fn_56909203

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
GPU2*0J 8? *^
fYRW
U__inference_batch_normalization_107_layer_call_and_return_conditional_losses_56908405w
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
?	
?
:__inference_batch_normalization_107_layer_call_fn_56909177

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
GPU2*0J 8? *^
fYRW
U__inference_batch_normalization_107_layer_call_and_return_conditional_losses_56908258?
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
?1
?
$__inference__traced_restore_56909432
file_prefix<
"assignvariableop_conv2d_106_kernel:>
0assignvariableop_1_batch_normalization_106_gamma:=
/assignvariableop_2_batch_normalization_106_beta:D
6assignvariableop_3_batch_normalization_106_moving_mean:H
:assignvariableop_4_batch_normalization_106_moving_variance:>
$assignvariableop_5_conv2d_107_kernel:>
0assignvariableop_6_batch_normalization_107_gamma:=
/assignvariableop_7_batch_normalization_107_beta:D
6assignvariableop_8_batch_normalization_107_moving_mean:H
:assignvariableop_9_batch_normalization_107_moving_variance:5
#assignvariableop_10_dense_53_kernel:d@
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
AssignVariableOpAssignVariableOp"assignvariableop_conv2d_106_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp0assignvariableop_1_batch_normalization_106_gammaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_106_betaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp6assignvariableop_3_batch_normalization_106_moving_meanIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp:assignvariableop_4_batch_normalization_106_moving_varianceIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp$assignvariableop_5_conv2d_107_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp0assignvariableop_6_batch_normalization_107_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp/assignvariableop_7_batch_normalization_107_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp6assignvariableop_8_batch_normalization_107_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp:assignvariableop_9_batch_normalization_107_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_53_kernelIdentity_10:output:0"/device:CPU:0*
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
?/
?
K__inference_sequential_53_layer_call_and_return_conditional_losses_56908671

inputs-
conv2d_106_56908638:.
 batch_normalization_106_56908642:.
 batch_normalization_106_56908644:.
 batch_normalization_106_56908646:.
 batch_normalization_106_56908648:-
conv2d_107_56908652:.
 batch_normalization_107_56908656:.
 batch_normalization_107_56908658:.
 batch_normalization_107_56908660:.
 batch_normalization_107_56908662:#
dense_53_56908667:d@
identity??/batch_normalization_106/StatefulPartitionedCall?/batch_normalization_107/StatefulPartitionedCall?"conv2d_106/StatefulPartitionedCall?"conv2d_107/StatefulPartitionedCall? dense_53/StatefulPartitionedCall?
"conv2d_106/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_106_56908638*
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
GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_106_layer_call_and_return_conditional_losses_56908326?
leaky_re_lu_106/PartitionedCallPartitionedCall+conv2d_106/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *V
fQRO
M__inference_leaky_re_lu_106_layer_call_and_return_conditional_losses_56908335?
/batch_normalization_106/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_106/PartitionedCall:output:0 batch_normalization_106_56908642 batch_normalization_106_56908644 batch_normalization_106_56908646 batch_normalization_106_56908648*
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
GPU2*0J 8? *^
fYRW
U__inference_batch_normalization_106_layer_call_and_return_conditional_losses_56908581?
!max_pooling2d_106/PartitionedCallPartitionedCall8batch_normalization_106/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *X
fSRQ
O__inference_max_pooling2d_106_layer_call_and_return_conditional_losses_56908368?
"conv2d_107/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_106/PartitionedCall:output:0conv2d_107_56908652*
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
GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_107_layer_call_and_return_conditional_losses_56908377?
leaky_re_lu_107/PartitionedCallPartitionedCall+conv2d_107/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *V
fQRO
M__inference_leaky_re_lu_107_layer_call_and_return_conditional_losses_56908386?
/batch_normalization_107/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_107/PartitionedCall:output:0 batch_normalization_107_56908656 batch_normalization_107_56908658 batch_normalization_107_56908660 batch_normalization_107_56908662*
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
GPU2*0J 8? *^
fYRW
U__inference_batch_normalization_107_layer_call_and_return_conditional_losses_56908518?
!max_pooling2d_107/PartitionedCallPartitionedCall8batch_normalization_107/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *X
fSRQ
O__inference_max_pooling2d_107_layer_call_and_return_conditional_losses_56908419?
flatten_53/PartitionedCallPartitionedCall*max_pooling2d_107/PartitionedCall:output:0*
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
H__inference_flatten_53_layer_call_and_return_conditional_losses_56908427?
 dense_53/StatefulPartitionedCallStatefulPartitionedCall#flatten_53/PartitionedCall:output:0dense_53_56908667*
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
F__inference_dense_53_layer_call_and_return_conditional_losses_56908436x
IdentityIdentity)dense_53/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp0^batch_normalization_106/StatefulPartitionedCall0^batch_normalization_107/StatefulPartitionedCall#^conv2d_106/StatefulPartitionedCall#^conv2d_107/StatefulPartitionedCall!^dense_53/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????: : : : : : : : : : : 2b
/batch_normalization_106/StatefulPartitionedCall/batch_normalization_106/StatefulPartitionedCall2b
/batch_normalization_107/StatefulPartitionedCall/batch_normalization_107/StatefulPartitionedCall2H
"conv2d_106/StatefulPartitionedCall"conv2d_106/StatefulPartitionedCall2H
"conv2d_107/StatefulPartitionedCall"conv2d_107/StatefulPartitionedCall2D
 dense_53/StatefulPartitionedCall dense_53/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
U__inference_batch_normalization_106_layer_call_and_return_conditional_losses_56908182

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
?
?
F__inference_dense_53_layer_call_and_return_conditional_losses_56908436

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
?
P
4__inference_max_pooling2d_106_layer_call_fn_56909125

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
GPU2*0J 8? *X
fSRQ
O__inference_max_pooling2d_106_layer_call_and_return_conditional_losses_56908233?
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
P
4__inference_max_pooling2d_106_layer_call_fn_56909130

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
GPU2*0J 8? *X
fSRQ
O__inference_max_pooling2d_106_layer_call_and_return_conditional_losses_56908368h
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
?
?
U__inference_batch_normalization_107_layer_call_and_return_conditional_losses_56909252

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
i
M__inference_leaky_re_lu_107_layer_call_and_return_conditional_losses_56908386

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
?
k
O__inference_max_pooling2d_107_layer_call_and_return_conditional_losses_56908419

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
?
?
U__inference_batch_normalization_106_layer_call_and_return_conditional_losses_56909084

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
U__inference_batch_normalization_106_layer_call_and_return_conditional_losses_56908354

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
?
?
U__inference_batch_normalization_106_layer_call_and_return_conditional_losses_56909102

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
?
?
U__inference_batch_normalization_107_layer_call_and_return_conditional_losses_56908258

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
?
0__inference_sequential_53_layer_call_fn_56908723
conv2d_106_input!
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
StatefulPartitionedCallStatefulPartitionedCallconv2d_106_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
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
K__inference_sequential_53_layer_call_and_return_conditional_losses_56908671o
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
StatefulPartitionedCallStatefulPartitionedCall:a ]
/
_output_shapes
:?????????
*
_user_specified_nameconv2d_106_input
?	
?
:__inference_batch_normalization_107_layer_call_fn_56909190

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
GPU2*0J 8? *^
fYRW
U__inference_batch_normalization_107_layer_call_and_return_conditional_losses_56908289?
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
?
k
O__inference_max_pooling2d_107_layer_call_and_return_conditional_losses_56908309

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
U__inference_batch_normalization_107_layer_call_and_return_conditional_losses_56908405

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
?
?
:__inference_batch_normalization_106_layer_call_fn_56909048

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
GPU2*0J 8? *^
fYRW
U__inference_batch_normalization_106_layer_call_and_return_conditional_losses_56908581w
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
?
N
2__inference_leaky_re_lu_107_layer_call_fn_56909159

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
GPU2*0J 8? *V
fQRO
M__inference_leaky_re_lu_107_layer_call_and_return_conditional_losses_56908386h
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
?
?
H__inference_conv2d_107_layer_call_and_return_conditional_losses_56909154

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
?

+__inference_dense_53_layer_call_fn_56909326

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
F__inference_dense_53_layer_call_and_return_conditional_losses_56908436o
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
?
?
U__inference_batch_normalization_107_layer_call_and_return_conditional_losses_56908518

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
?:
?

K__inference_sequential_53_layer_call_and_return_conditional_losses_56908972

inputsC
)conv2d_106_conv2d_readvariableop_resource:=
/batch_normalization_106_readvariableop_resource:?
1batch_normalization_106_readvariableop_1_resource:N
@batch_normalization_106_fusedbatchnormv3_readvariableop_resource:P
Bbatch_normalization_106_fusedbatchnormv3_readvariableop_1_resource:C
)conv2d_107_conv2d_readvariableop_resource:=
/batch_normalization_107_readvariableop_resource:?
1batch_normalization_107_readvariableop_1_resource:N
@batch_normalization_107_fusedbatchnormv3_readvariableop_resource:P
Bbatch_normalization_107_fusedbatchnormv3_readvariableop_1_resource:9
'dense_53_matmul_readvariableop_resource:d@
identity??7batch_normalization_106/FusedBatchNormV3/ReadVariableOp?9batch_normalization_106/FusedBatchNormV3/ReadVariableOp_1?&batch_normalization_106/ReadVariableOp?(batch_normalization_106/ReadVariableOp_1?7batch_normalization_107/FusedBatchNormV3/ReadVariableOp?9batch_normalization_107/FusedBatchNormV3/ReadVariableOp_1?&batch_normalization_107/ReadVariableOp?(batch_normalization_107/ReadVariableOp_1? conv2d_106/Conv2D/ReadVariableOp? conv2d_107/Conv2D/ReadVariableOp?dense_53/MatMul/ReadVariableOp?
 conv2d_106/Conv2D/ReadVariableOpReadVariableOp)conv2d_106_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_106/Conv2DConv2Dinputs(conv2d_106/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
leaky_re_lu_106/LeakyRelu	LeakyReluconv2d_106/Conv2D:output:0*/
_output_shapes
:?????????*
alpha%
?#<?
&batch_normalization_106/ReadVariableOpReadVariableOp/batch_normalization_106_readvariableop_resource*
_output_shapes
:*
dtype0?
(batch_normalization_106/ReadVariableOp_1ReadVariableOp1batch_normalization_106_readvariableop_1_resource*
_output_shapes
:*
dtype0?
7batch_normalization_106/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_106_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
9batch_normalization_106/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_106_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
(batch_normalization_106/FusedBatchNormV3FusedBatchNormV3'leaky_re_lu_106/LeakyRelu:activations:0.batch_normalization_106/ReadVariableOp:value:00batch_normalization_106/ReadVariableOp_1:value:0?batch_normalization_106/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_106/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
is_training( ?
max_pooling2d_106/MaxPoolMaxPool,batch_normalization_106/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
?
 conv2d_107/Conv2D/ReadVariableOpReadVariableOp)conv2d_107_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_107/Conv2DConv2D"max_pooling2d_106/MaxPool:output:0(conv2d_107/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
leaky_re_lu_107/LeakyRelu	LeakyReluconv2d_107/Conv2D:output:0*/
_output_shapes
:?????????*
alpha%
?#<?
&batch_normalization_107/ReadVariableOpReadVariableOp/batch_normalization_107_readvariableop_resource*
_output_shapes
:*
dtype0?
(batch_normalization_107/ReadVariableOp_1ReadVariableOp1batch_normalization_107_readvariableop_1_resource*
_output_shapes
:*
dtype0?
7batch_normalization_107/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_107_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
9batch_normalization_107/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_107_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
(batch_normalization_107/FusedBatchNormV3FusedBatchNormV3'leaky_re_lu_107/LeakyRelu:activations:0.batch_normalization_107/ReadVariableOp:value:00batch_normalization_107/ReadVariableOp_1:value:0?batch_normalization_107/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_107/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
is_training( ?
max_pooling2d_107/MaxPoolMaxPool,batch_normalization_107/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
a
flatten_53/ConstConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
flatten_53/ReshapeReshape"max_pooling2d_107/MaxPool:output:0flatten_53/Const:output:0*
T0*'
_output_shapes
:?????????d?
dense_53/MatMul/ReadVariableOpReadVariableOp'dense_53_matmul_readvariableop_resource*
_output_shapes

:d@*
dtype0?
dense_53/MatMulMatMulflatten_53/Reshape:output:0&dense_53/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@h
IdentityIdentitydense_53/MatMul:product:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp8^batch_normalization_106/FusedBatchNormV3/ReadVariableOp:^batch_normalization_106/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_106/ReadVariableOp)^batch_normalization_106/ReadVariableOp_18^batch_normalization_107/FusedBatchNormV3/ReadVariableOp:^batch_normalization_107/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_107/ReadVariableOp)^batch_normalization_107/ReadVariableOp_1!^conv2d_106/Conv2D/ReadVariableOp!^conv2d_107/Conv2D/ReadVariableOp^dense_53/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????: : : : : : : : : : : 2r
7batch_normalization_106/FusedBatchNormV3/ReadVariableOp7batch_normalization_106/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_106/FusedBatchNormV3/ReadVariableOp_19batch_normalization_106/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_106/ReadVariableOp&batch_normalization_106/ReadVariableOp2T
(batch_normalization_106/ReadVariableOp_1(batch_normalization_106/ReadVariableOp_12r
7batch_normalization_107/FusedBatchNormV3/ReadVariableOp7batch_normalization_107/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_107/FusedBatchNormV3/ReadVariableOp_19batch_normalization_107/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_107/ReadVariableOp&batch_normalization_107/ReadVariableOp2T
(batch_normalization_107/ReadVariableOp_1(batch_normalization_107/ReadVariableOp_12D
 conv2d_106/Conv2D/ReadVariableOp conv2d_106/Conv2D/ReadVariableOp2D
 conv2d_107/Conv2D/ReadVariableOp conv2d_107/Conv2D/ReadVariableOp2@
dense_53/MatMul/ReadVariableOpdense_53/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
0__inference_sequential_53_layer_call_fn_56908878

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
K__inference_sequential_53_layer_call_and_return_conditional_losses_56908671o
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
?
?
U__inference_batch_normalization_107_layer_call_and_return_conditional_losses_56909234

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
?
k
O__inference_max_pooling2d_106_layer_call_and_return_conditional_losses_56909135

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
?:
?

K__inference_sequential_53_layer_call_and_return_conditional_losses_56908925

inputsC
)conv2d_106_conv2d_readvariableop_resource:=
/batch_normalization_106_readvariableop_resource:?
1batch_normalization_106_readvariableop_1_resource:N
@batch_normalization_106_fusedbatchnormv3_readvariableop_resource:P
Bbatch_normalization_106_fusedbatchnormv3_readvariableop_1_resource:C
)conv2d_107_conv2d_readvariableop_resource:=
/batch_normalization_107_readvariableop_resource:?
1batch_normalization_107_readvariableop_1_resource:N
@batch_normalization_107_fusedbatchnormv3_readvariableop_resource:P
Bbatch_normalization_107_fusedbatchnormv3_readvariableop_1_resource:9
'dense_53_matmul_readvariableop_resource:d@
identity??7batch_normalization_106/FusedBatchNormV3/ReadVariableOp?9batch_normalization_106/FusedBatchNormV3/ReadVariableOp_1?&batch_normalization_106/ReadVariableOp?(batch_normalization_106/ReadVariableOp_1?7batch_normalization_107/FusedBatchNormV3/ReadVariableOp?9batch_normalization_107/FusedBatchNormV3/ReadVariableOp_1?&batch_normalization_107/ReadVariableOp?(batch_normalization_107/ReadVariableOp_1? conv2d_106/Conv2D/ReadVariableOp? conv2d_107/Conv2D/ReadVariableOp?dense_53/MatMul/ReadVariableOp?
 conv2d_106/Conv2D/ReadVariableOpReadVariableOp)conv2d_106_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_106/Conv2DConv2Dinputs(conv2d_106/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
leaky_re_lu_106/LeakyRelu	LeakyReluconv2d_106/Conv2D:output:0*/
_output_shapes
:?????????*
alpha%
?#<?
&batch_normalization_106/ReadVariableOpReadVariableOp/batch_normalization_106_readvariableop_resource*
_output_shapes
:*
dtype0?
(batch_normalization_106/ReadVariableOp_1ReadVariableOp1batch_normalization_106_readvariableop_1_resource*
_output_shapes
:*
dtype0?
7batch_normalization_106/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_106_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
9batch_normalization_106/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_106_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
(batch_normalization_106/FusedBatchNormV3FusedBatchNormV3'leaky_re_lu_106/LeakyRelu:activations:0.batch_normalization_106/ReadVariableOp:value:00batch_normalization_106/ReadVariableOp_1:value:0?batch_normalization_106/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_106/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
is_training( ?
max_pooling2d_106/MaxPoolMaxPool,batch_normalization_106/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
?
 conv2d_107/Conv2D/ReadVariableOpReadVariableOp)conv2d_107_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_107/Conv2DConv2D"max_pooling2d_106/MaxPool:output:0(conv2d_107/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
leaky_re_lu_107/LeakyRelu	LeakyReluconv2d_107/Conv2D:output:0*/
_output_shapes
:?????????*
alpha%
?#<?
&batch_normalization_107/ReadVariableOpReadVariableOp/batch_normalization_107_readvariableop_resource*
_output_shapes
:*
dtype0?
(batch_normalization_107/ReadVariableOp_1ReadVariableOp1batch_normalization_107_readvariableop_1_resource*
_output_shapes
:*
dtype0?
7batch_normalization_107/FusedBatchNormV3/ReadVariableOpReadVariableOp@batch_normalization_107_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
9batch_normalization_107/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpBbatch_normalization_107_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
(batch_normalization_107/FusedBatchNormV3FusedBatchNormV3'leaky_re_lu_107/LeakyRelu:activations:0.batch_normalization_107/ReadVariableOp:value:00batch_normalization_107/ReadVariableOp_1:value:0?batch_normalization_107/FusedBatchNormV3/ReadVariableOp:value:0Abatch_normalization_107/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
is_training( ?
max_pooling2d_107/MaxPoolMaxPool,batch_normalization_107/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
a
flatten_53/ConstConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
flatten_53/ReshapeReshape"max_pooling2d_107/MaxPool:output:0flatten_53/Const:output:0*
T0*'
_output_shapes
:?????????d?
dense_53/MatMul/ReadVariableOpReadVariableOp'dense_53_matmul_readvariableop_resource*
_output_shapes

:d@*
dtype0?
dense_53/MatMulMatMulflatten_53/Reshape:output:0&dense_53/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@h
IdentityIdentitydense_53/MatMul:product:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp8^batch_normalization_106/FusedBatchNormV3/ReadVariableOp:^batch_normalization_106/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_106/ReadVariableOp)^batch_normalization_106/ReadVariableOp_18^batch_normalization_107/FusedBatchNormV3/ReadVariableOp:^batch_normalization_107/FusedBatchNormV3/ReadVariableOp_1'^batch_normalization_107/ReadVariableOp)^batch_normalization_107/ReadVariableOp_1!^conv2d_106/Conv2D/ReadVariableOp!^conv2d_107/Conv2D/ReadVariableOp^dense_53/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????: : : : : : : : : : : 2r
7batch_normalization_106/FusedBatchNormV3/ReadVariableOp7batch_normalization_106/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_106/FusedBatchNormV3/ReadVariableOp_19batch_normalization_106/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_106/ReadVariableOp&batch_normalization_106/ReadVariableOp2T
(batch_normalization_106/ReadVariableOp_1(batch_normalization_106/ReadVariableOp_12r
7batch_normalization_107/FusedBatchNormV3/ReadVariableOp7batch_normalization_107/FusedBatchNormV3/ReadVariableOp2v
9batch_normalization_107/FusedBatchNormV3/ReadVariableOp_19batch_normalization_107/FusedBatchNormV3/ReadVariableOp_12P
&batch_normalization_107/ReadVariableOp&batch_normalization_107/ReadVariableOp2T
(batch_normalization_107/ReadVariableOp_1(batch_normalization_107/ReadVariableOp_12D
 conv2d_106/Conv2D/ReadVariableOp conv2d_106/Conv2D/ReadVariableOp2D
 conv2d_107/Conv2D/ReadVariableOp conv2d_107/Conv2D/ReadVariableOp2@
dense_53/MatMul/ReadVariableOpdense_53/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
k
O__inference_max_pooling2d_106_layer_call_and_return_conditional_losses_56908368

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
?
?
:__inference_batch_normalization_106_layer_call_fn_56909035

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
GPU2*0J 8? *^
fYRW
U__inference_batch_normalization_106_layer_call_and_return_conditional_losses_56908354w
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
?
i
M__inference_leaky_re_lu_106_layer_call_and_return_conditional_losses_56908996

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
:__inference_batch_normalization_106_layer_call_fn_56909022

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
GPU2*0J 8? *^
fYRW
U__inference_batch_normalization_106_layer_call_and_return_conditional_losses_56908213?
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
?/
?
K__inference_sequential_53_layer_call_and_return_conditional_losses_56908795
conv2d_106_input-
conv2d_106_56908762:.
 batch_normalization_106_56908766:.
 batch_normalization_106_56908768:.
 batch_normalization_106_56908770:.
 batch_normalization_106_56908772:-
conv2d_107_56908776:.
 batch_normalization_107_56908780:.
 batch_normalization_107_56908782:.
 batch_normalization_107_56908784:.
 batch_normalization_107_56908786:#
dense_53_56908791:d@
identity??/batch_normalization_106/StatefulPartitionedCall?/batch_normalization_107/StatefulPartitionedCall?"conv2d_106/StatefulPartitionedCall?"conv2d_107/StatefulPartitionedCall? dense_53/StatefulPartitionedCall?
"conv2d_106/StatefulPartitionedCallStatefulPartitionedCallconv2d_106_inputconv2d_106_56908762*
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
GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_106_layer_call_and_return_conditional_losses_56908326?
leaky_re_lu_106/PartitionedCallPartitionedCall+conv2d_106/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *V
fQRO
M__inference_leaky_re_lu_106_layer_call_and_return_conditional_losses_56908335?
/batch_normalization_106/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_106/PartitionedCall:output:0 batch_normalization_106_56908766 batch_normalization_106_56908768 batch_normalization_106_56908770 batch_normalization_106_56908772*
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
GPU2*0J 8? *^
fYRW
U__inference_batch_normalization_106_layer_call_and_return_conditional_losses_56908581?
!max_pooling2d_106/PartitionedCallPartitionedCall8batch_normalization_106/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *X
fSRQ
O__inference_max_pooling2d_106_layer_call_and_return_conditional_losses_56908368?
"conv2d_107/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_106/PartitionedCall:output:0conv2d_107_56908776*
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
GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_107_layer_call_and_return_conditional_losses_56908377?
leaky_re_lu_107/PartitionedCallPartitionedCall+conv2d_107/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *V
fQRO
M__inference_leaky_re_lu_107_layer_call_and_return_conditional_losses_56908386?
/batch_normalization_107/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_107/PartitionedCall:output:0 batch_normalization_107_56908780 batch_normalization_107_56908782 batch_normalization_107_56908784 batch_normalization_107_56908786*
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
GPU2*0J 8? *^
fYRW
U__inference_batch_normalization_107_layer_call_and_return_conditional_losses_56908518?
!max_pooling2d_107/PartitionedCallPartitionedCall8batch_normalization_107/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *X
fSRQ
O__inference_max_pooling2d_107_layer_call_and_return_conditional_losses_56908419?
flatten_53/PartitionedCallPartitionedCall*max_pooling2d_107/PartitionedCall:output:0*
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
H__inference_flatten_53_layer_call_and_return_conditional_losses_56908427?
 dense_53/StatefulPartitionedCallStatefulPartitionedCall#flatten_53/PartitionedCall:output:0dense_53_56908791*
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
F__inference_dense_53_layer_call_and_return_conditional_losses_56908436x
IdentityIdentity)dense_53/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp0^batch_normalization_106/StatefulPartitionedCall0^batch_normalization_107/StatefulPartitionedCall#^conv2d_106/StatefulPartitionedCall#^conv2d_107/StatefulPartitionedCall!^dense_53/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????: : : : : : : : : : : 2b
/batch_normalization_106/StatefulPartitionedCall/batch_normalization_106/StatefulPartitionedCall2b
/batch_normalization_107/StatefulPartitionedCall/batch_normalization_107/StatefulPartitionedCall2H
"conv2d_106/StatefulPartitionedCall"conv2d_106/StatefulPartitionedCall2H
"conv2d_107/StatefulPartitionedCall"conv2d_107/StatefulPartitionedCall2D
 dense_53/StatefulPartitionedCall dense_53/StatefulPartitionedCall:a ]
/
_output_shapes
:?????????
*
_user_specified_nameconv2d_106_input
?
i
M__inference_leaky_re_lu_106_layer_call_and_return_conditional_losses_56908335

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
?
N
2__inference_leaky_re_lu_106_layer_call_fn_56908991

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
GPU2*0J 8? *V
fQRO
M__inference_leaky_re_lu_106_layer_call_and_return_conditional_losses_56908335h
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
?	
?
:__inference_batch_normalization_106_layer_call_fn_56909009

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
GPU2*0J 8? *^
fYRW
U__inference_batch_normalization_106_layer_call_and_return_conditional_losses_56908182?
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
?
I
-__inference_flatten_53_layer_call_fn_56909313

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
H__inference_flatten_53_layer_call_and_return_conditional_losses_56908427`
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
?
?
U__inference_batch_normalization_106_layer_call_and_return_conditional_losses_56908213

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
?
?
H__inference_conv2d_106_layer_call_and_return_conditional_losses_56908326

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
?
?
U__inference_batch_normalization_107_layer_call_and_return_conditional_losses_56909270

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
?
F__inference_dense_53_layer_call_and_return_conditional_losses_56909333

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
?
k
O__inference_max_pooling2d_107_layer_call_and_return_conditional_losses_56909308

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
?$
?
!__inference__traced_save_56909389
file_prefix0
,savev2_conv2d_106_kernel_read_readvariableop<
8savev2_batch_normalization_106_gamma_read_readvariableop;
7savev2_batch_normalization_106_beta_read_readvariableopB
>savev2_batch_normalization_106_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_106_moving_variance_read_readvariableop0
,savev2_conv2d_107_kernel_read_readvariableop<
8savev2_batch_normalization_107_gamma_read_readvariableop;
7savev2_batch_normalization_107_beta_read_readvariableopB
>savev2_batch_normalization_107_moving_mean_read_readvariableopF
Bsavev2_batch_normalization_107_moving_variance_read_readvariableop.
*savev2_dense_53_kernel_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_conv2d_106_kernel_read_readvariableop8savev2_batch_normalization_106_gamma_read_readvariableop7savev2_batch_normalization_106_beta_read_readvariableop>savev2_batch_normalization_106_moving_mean_read_readvariableopBsavev2_batch_normalization_106_moving_variance_read_readvariableop,savev2_conv2d_107_kernel_read_readvariableop8savev2_batch_normalization_107_gamma_read_readvariableop7savev2_batch_normalization_107_beta_read_readvariableop>savev2_batch_normalization_107_moving_mean_read_readvariableopBsavev2_batch_normalization_107_moving_variance_read_readvariableop*savev2_dense_53_kernel_read_readvariableopsavev2_const"/device:CPU:0*
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
?
d
H__inference_flatten_53_layer_call_and_return_conditional_losses_56908427

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
?
P
4__inference_max_pooling2d_107_layer_call_fn_56909293

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
GPU2*0J 8? *X
fSRQ
O__inference_max_pooling2d_107_layer_call_and_return_conditional_losses_56908309?
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
?
?
H__inference_conv2d_107_layer_call_and_return_conditional_losses_56908377

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
?
?
H__inference_conv2d_106_layer_call_and_return_conditional_losses_56908986

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
i
M__inference_leaky_re_lu_107_layer_call_and_return_conditional_losses_56909164

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
?
?
U__inference_batch_normalization_107_layer_call_and_return_conditional_losses_56909288

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
d
H__inference_flatten_53_layer_call_and_return_conditional_losses_56909319

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
?
0__inference_sequential_53_layer_call_fn_56908466
conv2d_106_input!
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
StatefulPartitionedCallStatefulPartitionedCallconv2d_106_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
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
K__inference_sequential_53_layer_call_and_return_conditional_losses_56908441o
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
StatefulPartitionedCallStatefulPartitionedCall:a ]
/
_output_shapes
:?????????
*
_user_specified_nameconv2d_106_input
?F
?
#__inference__wrapped_model_56908160
conv2d_106_inputQ
7sequential_53_conv2d_106_conv2d_readvariableop_resource:K
=sequential_53_batch_normalization_106_readvariableop_resource:M
?sequential_53_batch_normalization_106_readvariableop_1_resource:\
Nsequential_53_batch_normalization_106_fusedbatchnormv3_readvariableop_resource:^
Psequential_53_batch_normalization_106_fusedbatchnormv3_readvariableop_1_resource:Q
7sequential_53_conv2d_107_conv2d_readvariableop_resource:K
=sequential_53_batch_normalization_107_readvariableop_resource:M
?sequential_53_batch_normalization_107_readvariableop_1_resource:\
Nsequential_53_batch_normalization_107_fusedbatchnormv3_readvariableop_resource:^
Psequential_53_batch_normalization_107_fusedbatchnormv3_readvariableop_1_resource:G
5sequential_53_dense_53_matmul_readvariableop_resource:d@
identity??Esequential_53/batch_normalization_106/FusedBatchNormV3/ReadVariableOp?Gsequential_53/batch_normalization_106/FusedBatchNormV3/ReadVariableOp_1?4sequential_53/batch_normalization_106/ReadVariableOp?6sequential_53/batch_normalization_106/ReadVariableOp_1?Esequential_53/batch_normalization_107/FusedBatchNormV3/ReadVariableOp?Gsequential_53/batch_normalization_107/FusedBatchNormV3/ReadVariableOp_1?4sequential_53/batch_normalization_107/ReadVariableOp?6sequential_53/batch_normalization_107/ReadVariableOp_1?.sequential_53/conv2d_106/Conv2D/ReadVariableOp?.sequential_53/conv2d_107/Conv2D/ReadVariableOp?,sequential_53/dense_53/MatMul/ReadVariableOp?
.sequential_53/conv2d_106/Conv2D/ReadVariableOpReadVariableOp7sequential_53_conv2d_106_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential_53/conv2d_106/Conv2DConv2Dconv2d_106_input6sequential_53/conv2d_106/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
'sequential_53/leaky_re_lu_106/LeakyRelu	LeakyRelu(sequential_53/conv2d_106/Conv2D:output:0*/
_output_shapes
:?????????*
alpha%
?#<?
4sequential_53/batch_normalization_106/ReadVariableOpReadVariableOp=sequential_53_batch_normalization_106_readvariableop_resource*
_output_shapes
:*
dtype0?
6sequential_53/batch_normalization_106/ReadVariableOp_1ReadVariableOp?sequential_53_batch_normalization_106_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Esequential_53/batch_normalization_106/FusedBatchNormV3/ReadVariableOpReadVariableOpNsequential_53_batch_normalization_106_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Gsequential_53/batch_normalization_106/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPsequential_53_batch_normalization_106_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6sequential_53/batch_normalization_106/FusedBatchNormV3FusedBatchNormV35sequential_53/leaky_re_lu_106/LeakyRelu:activations:0<sequential_53/batch_normalization_106/ReadVariableOp:value:0>sequential_53/batch_normalization_106/ReadVariableOp_1:value:0Msequential_53/batch_normalization_106/FusedBatchNormV3/ReadVariableOp:value:0Osequential_53/batch_normalization_106/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
is_training( ?
'sequential_53/max_pooling2d_106/MaxPoolMaxPool:sequential_53/batch_normalization_106/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
?
.sequential_53/conv2d_107/Conv2D/ReadVariableOpReadVariableOp7sequential_53_conv2d_107_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
sequential_53/conv2d_107/Conv2DConv2D0sequential_53/max_pooling2d_106/MaxPool:output:06sequential_53/conv2d_107/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
?
'sequential_53/leaky_re_lu_107/LeakyRelu	LeakyRelu(sequential_53/conv2d_107/Conv2D:output:0*/
_output_shapes
:?????????*
alpha%
?#<?
4sequential_53/batch_normalization_107/ReadVariableOpReadVariableOp=sequential_53_batch_normalization_107_readvariableop_resource*
_output_shapes
:*
dtype0?
6sequential_53/batch_normalization_107/ReadVariableOp_1ReadVariableOp?sequential_53_batch_normalization_107_readvariableop_1_resource*
_output_shapes
:*
dtype0?
Esequential_53/batch_normalization_107/FusedBatchNormV3/ReadVariableOpReadVariableOpNsequential_53_batch_normalization_107_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
Gsequential_53/batch_normalization_107/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPsequential_53_batch_normalization_107_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6sequential_53/batch_normalization_107/FusedBatchNormV3FusedBatchNormV35sequential_53/leaky_re_lu_107/LeakyRelu:activations:0<sequential_53/batch_normalization_107/ReadVariableOp:value:0>sequential_53/batch_normalization_107/ReadVariableOp_1:value:0Msequential_53/batch_normalization_107/FusedBatchNormV3/ReadVariableOp:value:0Osequential_53/batch_normalization_107/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
is_training( ?
'sequential_53/max_pooling2d_107/MaxPoolMaxPool:sequential_53/batch_normalization_107/FusedBatchNormV3:y:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
o
sequential_53/flatten_53/ConstConst*
_output_shapes
:*
dtype0*
valueB"????d   ?
 sequential_53/flatten_53/ReshapeReshape0sequential_53/max_pooling2d_107/MaxPool:output:0'sequential_53/flatten_53/Const:output:0*
T0*'
_output_shapes
:?????????d?
,sequential_53/dense_53/MatMul/ReadVariableOpReadVariableOp5sequential_53_dense_53_matmul_readvariableop_resource*
_output_shapes

:d@*
dtype0?
sequential_53/dense_53/MatMulMatMul)sequential_53/flatten_53/Reshape:output:04sequential_53/dense_53/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@v
IdentityIdentity'sequential_53/dense_53/MatMul:product:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOpF^sequential_53/batch_normalization_106/FusedBatchNormV3/ReadVariableOpH^sequential_53/batch_normalization_106/FusedBatchNormV3/ReadVariableOp_15^sequential_53/batch_normalization_106/ReadVariableOp7^sequential_53/batch_normalization_106/ReadVariableOp_1F^sequential_53/batch_normalization_107/FusedBatchNormV3/ReadVariableOpH^sequential_53/batch_normalization_107/FusedBatchNormV3/ReadVariableOp_15^sequential_53/batch_normalization_107/ReadVariableOp7^sequential_53/batch_normalization_107/ReadVariableOp_1/^sequential_53/conv2d_106/Conv2D/ReadVariableOp/^sequential_53/conv2d_107/Conv2D/ReadVariableOp-^sequential_53/dense_53/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????: : : : : : : : : : : 2?
Esequential_53/batch_normalization_106/FusedBatchNormV3/ReadVariableOpEsequential_53/batch_normalization_106/FusedBatchNormV3/ReadVariableOp2?
Gsequential_53/batch_normalization_106/FusedBatchNormV3/ReadVariableOp_1Gsequential_53/batch_normalization_106/FusedBatchNormV3/ReadVariableOp_12l
4sequential_53/batch_normalization_106/ReadVariableOp4sequential_53/batch_normalization_106/ReadVariableOp2p
6sequential_53/batch_normalization_106/ReadVariableOp_16sequential_53/batch_normalization_106/ReadVariableOp_12?
Esequential_53/batch_normalization_107/FusedBatchNormV3/ReadVariableOpEsequential_53/batch_normalization_107/FusedBatchNormV3/ReadVariableOp2?
Gsequential_53/batch_normalization_107/FusedBatchNormV3/ReadVariableOp_1Gsequential_53/batch_normalization_107/FusedBatchNormV3/ReadVariableOp_12l
4sequential_53/batch_normalization_107/ReadVariableOp4sequential_53/batch_normalization_107/ReadVariableOp2p
6sequential_53/batch_normalization_107/ReadVariableOp_16sequential_53/batch_normalization_107/ReadVariableOp_12`
.sequential_53/conv2d_106/Conv2D/ReadVariableOp.sequential_53/conv2d_106/Conv2D/ReadVariableOp2`
.sequential_53/conv2d_107/Conv2D/ReadVariableOp.sequential_53/conv2d_107/Conv2D/ReadVariableOp2\
,sequential_53/dense_53/MatMul/ReadVariableOp,sequential_53/dense_53/MatMul/ReadVariableOp:a ]
/
_output_shapes
:?????????
*
_user_specified_nameconv2d_106_input
?
?
U__inference_batch_normalization_107_layer_call_and_return_conditional_losses_56908289

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

?
0__inference_sequential_53_layer_call_fn_56908851

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
K__inference_sequential_53_layer_call_and_return_conditional_losses_56908441o
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
?
?
-__inference_conv2d_106_layer_call_fn_56908979

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
GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_106_layer_call_and_return_conditional_losses_56908326w
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
?
?
-__inference_conv2d_107_layer_call_fn_56909147

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
GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_107_layer_call_and_return_conditional_losses_56908377w
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
?
?
U__inference_batch_normalization_106_layer_call_and_return_conditional_losses_56909120

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
?
?
:__inference_batch_normalization_107_layer_call_fn_56909216

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
GPU2*0J 8? *^
fYRW
U__inference_batch_normalization_107_layer_call_and_return_conditional_losses_56908518w
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
?
?
U__inference_batch_normalization_106_layer_call_and_return_conditional_losses_56908581

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
?

?
&__inference_signature_wrapper_56908824
conv2d_106_input!
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
StatefulPartitionedCallStatefulPartitionedCallconv2d_106_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
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
#__inference__wrapped_model_56908160o
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
StatefulPartitionedCallStatefulPartitionedCall:a ]
/
_output_shapes
:?????????
*
_user_specified_nameconv2d_106_input
?
P
4__inference_max_pooling2d_107_layer_call_fn_56909298

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
GPU2*0J 8? *X
fSRQ
O__inference_max_pooling2d_107_layer_call_and_return_conditional_losses_56908419h
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
?
?
U__inference_batch_normalization_106_layer_call_and_return_conditional_losses_56909066

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
K__inference_sequential_53_layer_call_and_return_conditional_losses_56908441

inputs-
conv2d_106_56908327:.
 batch_normalization_106_56908355:.
 batch_normalization_106_56908357:.
 batch_normalization_106_56908359:.
 batch_normalization_106_56908361:-
conv2d_107_56908378:.
 batch_normalization_107_56908406:.
 batch_normalization_107_56908408:.
 batch_normalization_107_56908410:.
 batch_normalization_107_56908412:#
dense_53_56908437:d@
identity??/batch_normalization_106/StatefulPartitionedCall?/batch_normalization_107/StatefulPartitionedCall?"conv2d_106/StatefulPartitionedCall?"conv2d_107/StatefulPartitionedCall? dense_53/StatefulPartitionedCall?
"conv2d_106/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_106_56908327*
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
GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_106_layer_call_and_return_conditional_losses_56908326?
leaky_re_lu_106/PartitionedCallPartitionedCall+conv2d_106/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *V
fQRO
M__inference_leaky_re_lu_106_layer_call_and_return_conditional_losses_56908335?
/batch_normalization_106/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_106/PartitionedCall:output:0 batch_normalization_106_56908355 batch_normalization_106_56908357 batch_normalization_106_56908359 batch_normalization_106_56908361*
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
GPU2*0J 8? *^
fYRW
U__inference_batch_normalization_106_layer_call_and_return_conditional_losses_56908354?
!max_pooling2d_106/PartitionedCallPartitionedCall8batch_normalization_106/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *X
fSRQ
O__inference_max_pooling2d_106_layer_call_and_return_conditional_losses_56908368?
"conv2d_107/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_106/PartitionedCall:output:0conv2d_107_56908378*
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
GPU2*0J 8? *Q
fLRJ
H__inference_conv2d_107_layer_call_and_return_conditional_losses_56908377?
leaky_re_lu_107/PartitionedCallPartitionedCall+conv2d_107/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *V
fQRO
M__inference_leaky_re_lu_107_layer_call_and_return_conditional_losses_56908386?
/batch_normalization_107/StatefulPartitionedCallStatefulPartitionedCall(leaky_re_lu_107/PartitionedCall:output:0 batch_normalization_107_56908406 batch_normalization_107_56908408 batch_normalization_107_56908410 batch_normalization_107_56908412*
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
GPU2*0J 8? *^
fYRW
U__inference_batch_normalization_107_layer_call_and_return_conditional_losses_56908405?
!max_pooling2d_107/PartitionedCallPartitionedCall8batch_normalization_107/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *X
fSRQ
O__inference_max_pooling2d_107_layer_call_and_return_conditional_losses_56908419?
flatten_53/PartitionedCallPartitionedCall*max_pooling2d_107/PartitionedCall:output:0*
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
H__inference_flatten_53_layer_call_and_return_conditional_losses_56908427?
 dense_53/StatefulPartitionedCallStatefulPartitionedCall#flatten_53/PartitionedCall:output:0dense_53_56908437*
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
F__inference_dense_53_layer_call_and_return_conditional_losses_56908436x
IdentityIdentity)dense_53/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@?
NoOpNoOp0^batch_normalization_106/StatefulPartitionedCall0^batch_normalization_107/StatefulPartitionedCall#^conv2d_106/StatefulPartitionedCall#^conv2d_107/StatefulPartitionedCall!^dense_53/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:?????????: : : : : : : : : : : 2b
/batch_normalization_106/StatefulPartitionedCall/batch_normalization_106/StatefulPartitionedCall2b
/batch_normalization_107/StatefulPartitionedCall/batch_normalization_107/StatefulPartitionedCall2H
"conv2d_106/StatefulPartitionedCall"conv2d_106/StatefulPartitionedCall2H
"conv2d_107/StatefulPartitionedCall"conv2d_107/StatefulPartitionedCall2D
 dense_53/StatefulPartitionedCall dense_53/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
k
O__inference_max_pooling2d_107_layer_call_and_return_conditional_losses_56909303

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
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
U
conv2d_106_inputA
"serving_default_conv2d_106_input:0?????????<
dense_530
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
+:)2conv2d_106/kernel
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
+:)2batch_normalization_106/gamma
*:(2batch_normalization_106/beta
3:1 (2#batch_normalization_106/moving_mean
7:5 (2'batch_normalization_106/moving_variance
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
+:)2conv2d_107/kernel
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
+:)2batch_normalization_107/gamma
*:(2batch_normalization_107/beta
3:1 (2#batch_normalization_107/moving_mean
7:5 (2'batch_normalization_107/moving_variance
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
!:d@2dense_53/kernel
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
0__inference_sequential_53_layer_call_fn_56908466
0__inference_sequential_53_layer_call_fn_56908851
0__inference_sequential_53_layer_call_fn_56908878
0__inference_sequential_53_layer_call_fn_56908723?
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
K__inference_sequential_53_layer_call_and_return_conditional_losses_56908925
K__inference_sequential_53_layer_call_and_return_conditional_losses_56908972
K__inference_sequential_53_layer_call_and_return_conditional_losses_56908759
K__inference_sequential_53_layer_call_and_return_conditional_losses_56908795?
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
#__inference__wrapped_model_56908160conv2d_106_input"?
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
-__inference_conv2d_106_layer_call_fn_56908979?
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
H__inference_conv2d_106_layer_call_and_return_conditional_losses_56908986?
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
2__inference_leaky_re_lu_106_layer_call_fn_56908991?
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
M__inference_leaky_re_lu_106_layer_call_and_return_conditional_losses_56908996?
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
:__inference_batch_normalization_106_layer_call_fn_56909009
:__inference_batch_normalization_106_layer_call_fn_56909022
:__inference_batch_normalization_106_layer_call_fn_56909035
:__inference_batch_normalization_106_layer_call_fn_56909048?
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
U__inference_batch_normalization_106_layer_call_and_return_conditional_losses_56909066
U__inference_batch_normalization_106_layer_call_and_return_conditional_losses_56909084
U__inference_batch_normalization_106_layer_call_and_return_conditional_losses_56909102
U__inference_batch_normalization_106_layer_call_and_return_conditional_losses_56909120?
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
4__inference_max_pooling2d_106_layer_call_fn_56909125
4__inference_max_pooling2d_106_layer_call_fn_56909130?
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
O__inference_max_pooling2d_106_layer_call_and_return_conditional_losses_56909135
O__inference_max_pooling2d_106_layer_call_and_return_conditional_losses_56909140?
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
-__inference_conv2d_107_layer_call_fn_56909147?
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
H__inference_conv2d_107_layer_call_and_return_conditional_losses_56909154?
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
2__inference_leaky_re_lu_107_layer_call_fn_56909159?
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
M__inference_leaky_re_lu_107_layer_call_and_return_conditional_losses_56909164?
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
:__inference_batch_normalization_107_layer_call_fn_56909177
:__inference_batch_normalization_107_layer_call_fn_56909190
:__inference_batch_normalization_107_layer_call_fn_56909203
:__inference_batch_normalization_107_layer_call_fn_56909216?
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
U__inference_batch_normalization_107_layer_call_and_return_conditional_losses_56909234
U__inference_batch_normalization_107_layer_call_and_return_conditional_losses_56909252
U__inference_batch_normalization_107_layer_call_and_return_conditional_losses_56909270
U__inference_batch_normalization_107_layer_call_and_return_conditional_losses_56909288?
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
4__inference_max_pooling2d_107_layer_call_fn_56909293
4__inference_max_pooling2d_107_layer_call_fn_56909298?
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
O__inference_max_pooling2d_107_layer_call_and_return_conditional_losses_56909303
O__inference_max_pooling2d_107_layer_call_and_return_conditional_losses_56909308?
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
-__inference_flatten_53_layer_call_fn_56909313?
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
H__inference_flatten_53_layer_call_and_return_conditional_losses_56909319?
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
+__inference_dense_53_layer_call_fn_56909326?
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
F__inference_dense_53_layer_call_and_return_conditional_losses_56909333?
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
&__inference_signature_wrapper_56908824conv2d_106_input"?
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
#__inference__wrapped_model_56908160?&0123@A?>
7?4
2?/
conv2d_106_input?????????
? "3?0
.
dense_53"?
dense_53?????????@?
U__inference_batch_normalization_106_layer_call_and_return_conditional_losses_56909066?M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
U__inference_batch_normalization_106_layer_call_and_return_conditional_losses_56909084?M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
U__inference_batch_normalization_106_layer_call_and_return_conditional_losses_56909102r;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
U__inference_batch_normalization_106_layer_call_and_return_conditional_losses_56909120r;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
:__inference_batch_normalization_106_layer_call_fn_56909009?M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
:__inference_batch_normalization_106_layer_call_fn_56909022?M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
:__inference_batch_normalization_106_layer_call_fn_56909035e;?8
1?.
(?%
inputs?????????
p 
? " ???????????
:__inference_batch_normalization_106_layer_call_fn_56909048e;?8
1?.
(?%
inputs?????????
p
? " ???????????
U__inference_batch_normalization_107_layer_call_and_return_conditional_losses_56909234?0123M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
U__inference_batch_normalization_107_layer_call_and_return_conditional_losses_56909252?0123M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
U__inference_batch_normalization_107_layer_call_and_return_conditional_losses_56909270r0123;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
U__inference_batch_normalization_107_layer_call_and_return_conditional_losses_56909288r0123;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
:__inference_batch_normalization_107_layer_call_fn_56909177?0123M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
:__inference_batch_normalization_107_layer_call_fn_56909190?0123M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
:__inference_batch_normalization_107_layer_call_fn_56909203e0123;?8
1?.
(?%
inputs?????????
p 
? " ???????????
:__inference_batch_normalization_107_layer_call_fn_56909216e0123;?8
1?.
(?%
inputs?????????
p
? " ???????????
H__inference_conv2d_106_layer_call_and_return_conditional_losses_56908986k7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
-__inference_conv2d_106_layer_call_fn_56908979^7?4
-?*
(?%
inputs?????????
? " ???????????
H__inference_conv2d_107_layer_call_and_return_conditional_losses_56909154k&7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
-__inference_conv2d_107_layer_call_fn_56909147^&7?4
-?*
(?%
inputs?????????
? " ???????????
F__inference_dense_53_layer_call_and_return_conditional_losses_56909333[@/?,
%?"
 ?
inputs?????????d
? "%?"
?
0?????????@
? }
+__inference_dense_53_layer_call_fn_56909326N@/?,
%?"
 ?
inputs?????????d
? "??????????@?
H__inference_flatten_53_layer_call_and_return_conditional_losses_56909319`7?4
-?*
(?%
inputs?????????
? "%?"
?
0?????????d
? ?
-__inference_flatten_53_layer_call_fn_56909313S7?4
-?*
(?%
inputs?????????
? "??????????d?
M__inference_leaky_re_lu_106_layer_call_and_return_conditional_losses_56908996h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
2__inference_leaky_re_lu_106_layer_call_fn_56908991[7?4
-?*
(?%
inputs?????????
? " ???????????
M__inference_leaky_re_lu_107_layer_call_and_return_conditional_losses_56909164h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
2__inference_leaky_re_lu_107_layer_call_fn_56909159[7?4
-?*
(?%
inputs?????????
? " ???????????
O__inference_max_pooling2d_106_layer_call_and_return_conditional_losses_56909135?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
O__inference_max_pooling2d_106_layer_call_and_return_conditional_losses_56909140h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
4__inference_max_pooling2d_106_layer_call_fn_56909125?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
4__inference_max_pooling2d_106_layer_call_fn_56909130[7?4
-?*
(?%
inputs?????????
? " ???????????
O__inference_max_pooling2d_107_layer_call_and_return_conditional_losses_56909303?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
O__inference_max_pooling2d_107_layer_call_and_return_conditional_losses_56909308h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
4__inference_max_pooling2d_107_layer_call_fn_56909293?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
4__inference_max_pooling2d_107_layer_call_fn_56909298[7?4
-?*
(?%
inputs?????????
? " ???????????
K__inference_sequential_53_layer_call_and_return_conditional_losses_56908759&0123@I?F
??<
2?/
conv2d_106_input?????????
p 

 
? "%?"
?
0?????????@
? ?
K__inference_sequential_53_layer_call_and_return_conditional_losses_56908795&0123@I?F
??<
2?/
conv2d_106_input?????????
p

 
? "%?"
?
0?????????@
? ?
K__inference_sequential_53_layer_call_and_return_conditional_losses_56908925u&0123@??<
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
K__inference_sequential_53_layer_call_and_return_conditional_losses_56908972u&0123@??<
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
0__inference_sequential_53_layer_call_fn_56908466r&0123@I?F
??<
2?/
conv2d_106_input?????????
p 

 
? "??????????@?
0__inference_sequential_53_layer_call_fn_56908723r&0123@I?F
??<
2?/
conv2d_106_input?????????
p

 
? "??????????@?
0__inference_sequential_53_layer_call_fn_56908851h&0123@??<
5?2
(?%
inputs?????????
p 

 
? "??????????@?
0__inference_sequential_53_layer_call_fn_56908878h&0123@??<
5?2
(?%
inputs?????????
p

 
? "??????????@?
&__inference_signature_wrapper_56908824?&0123@U?R
? 
K?H
F
conv2d_106_input2?/
conv2d_106_input?????????"3?0
.
dense_53"?
dense_53?????????@