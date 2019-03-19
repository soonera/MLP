% x为训练集矩阵17X2,17个样品，每个样品两个属性值
x=[
0.697	0.46
0.774	0.376
0.634	0.264
0.608	0.318
0.556	0.215
0.403	0.237
0.481	0.149
0.437	0.211
0.666	0.091
0.243	0.267
0.245	0.057
0.343	0.099
0.639	0.161
0.657	0.198
0.36	0.37
0.593	0.042
0.719	0.103
];

% y为训练集17个样品的标签值
y=[1;1;1;1;1;1;1;1;0;0;0;0;0;0;0;0;0];

%学习率为eta，设置为0.05
eta=0.05;
    
% 此MLP共三层，第一层输入层2个神经元，第二层隐藏层2个神经元，第三层输出层1个神经元。
% v_11为输入层第1个神经元与隐藏层第1个神经元连接的权值，v_12,v_21,v_22同理
% gamma_1为隐藏层第一个神经元的阈值，gamma_2同理
% omega_1为隐藏层第1个神经元与输出层神经元连接的权值，omega_2同理
% theta为输出层神经元连接的阈值
%下面为随机初始化v_11、v_12、v_21、v_22、gamma_1、gamma_2、omega_1、omega_2、theta

v_11=unifrnd (0,1); 
v_12=unifrnd (0,1); 
v_21=unifrnd (0,1); 
v_22=unifrnd (0,1); 

gamma_1=unifrnd (0,1); 
gamma_2=unifrnd (0,1); 

omega_1=unifrnd (0,1); 
omega_2=unifrnd (0,1); 

theta=unifrnd (0,1); 

%E为cost function 的值，此处初始化为0
E=0;

%Y为根据MLP得到的17个样本的标签值，这里均初始化为0
Y=zeros(17,1);



while 1

for j=1:1:17
    
x1=x(j,1);%x1为第j个样品的第1个属性值
x2=x(j,2);%x2为第j个样品的第2个属性值

alpha_1=v_11*x1+v_21*x2; %alpha_1为隐藏层第1个神经元的输入
alpha_2=v_21*x1+v_22*x2; %alpha_2为隐藏层第2个神经元的输入

b1=sigmoid(alpha_1-gamma_1);%b1为隐藏层第1个神经元的输出
b2=sigmoid(alpha_2-gamma_2);%b2为隐藏层第2个神经元的输出

beta=omega_1*b1+omega_2*b2;%beta为输出层神经元的输入
y_c=sigmoid(beta-theta);%y_c为输出层神经元的输出

Y(j)=y_c;%将第j个样品经过MLP的输出值记录在Y(j)中

%以下为根据BP算法更新各参数值
g=y_c*(1-y_c)*(y(j)-y_c);%g为cost function对输出层神经元的输入beta的导数的相反数

e1=b1*(1-b1)*omega_1*g;%e1为cost function对隐藏层第1个神经元的输入alpha_1的导数的相反数
e2=b2*(1-b2)*omega_1*g;%e2为cost function对隐藏层第2个神经元的输入alpha_2的导数的相反数

delta_omega_1=eta*g*b1;
delta_omega_2=eta*g*b2;

omega_1=omega_1+delta_omega_1;
omega_2=omega_2+delta_omega_2;

delta_v_11=eta*e1*x1;
delta_v_12=eta*e2*x1;
delta_v_21=eta*e1*x2;
delta_v_22=eta*e2*x2;

v_11=v_11+delta_v_11;
v_12=v_12+delta_v_12;
v_21=v_21+delta_v_21;
v_22=v_22+delta_v_22;

delta_theta=-1*eta*g;
theta=theta+delta_theta;

delta_gamma_1=-1*eta*e1;
delta_gamma_2=-1*eta*e2;

gamma_1=gamma_1+delta_gamma_1;
gamma_2=gamma_2+delta_gamma_2;
%以上各句为根据链式法则更新参数值

E=E+0.5*(y_c-y(j)).^2;%计算cost function

end

if E<0.0001 %cost function<0.0001,就停止训练，否则继续依次根据17个样品更新参数值
    break;
end

E=0; %所有样品更新一次之后，E置0

end


%定义sigmoid函数
function output = sigmoid(x)
output =1./(1+exp(-x));
end







