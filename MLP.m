% xΪѵ��������17X2,17����Ʒ��ÿ����Ʒ��������ֵ
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

% yΪѵ����17����Ʒ�ı�ǩֵ
y=[1;1;1;1;1;1;1;1;0;0;0;0;0;0;0;0;0];

%ѧϰ��Ϊeta������Ϊ0.05
eta=0.05;
    
% ��MLP�����㣬��һ�������2����Ԫ���ڶ������ز�2����Ԫ�������������1����Ԫ��
% v_11Ϊ������1����Ԫ�����ز��1����Ԫ���ӵ�Ȩֵ��v_12,v_21,v_22ͬ��
% gamma_1Ϊ���ز��һ����Ԫ����ֵ��gamma_2ͬ��
% omega_1Ϊ���ز��1����Ԫ���������Ԫ���ӵ�Ȩֵ��omega_2ͬ��
% thetaΪ�������Ԫ���ӵ���ֵ
%����Ϊ�����ʼ��v_11��v_12��v_21��v_22��gamma_1��gamma_2��omega_1��omega_2��theta

v_11=unifrnd (0,1); 
v_12=unifrnd (0,1); 
v_21=unifrnd (0,1); 
v_22=unifrnd (0,1); 

gamma_1=unifrnd (0,1); 
gamma_2=unifrnd (0,1); 

omega_1=unifrnd (0,1); 
omega_2=unifrnd (0,1); 

theta=unifrnd (0,1); 

%EΪcost function ��ֵ���˴���ʼ��Ϊ0
E=0;

%YΪ����MLP�õ���17�������ı�ǩֵ���������ʼ��Ϊ0
Y=zeros(17,1);



while 1

for j=1:1:17
    
x1=x(j,1);%x1Ϊ��j����Ʒ�ĵ�1������ֵ
x2=x(j,2);%x2Ϊ��j����Ʒ�ĵ�2������ֵ

alpha_1=v_11*x1+v_21*x2; %alpha_1Ϊ���ز��1����Ԫ������
alpha_2=v_21*x1+v_22*x2; %alpha_2Ϊ���ز��2����Ԫ������

b1=sigmoid(alpha_1-gamma_1);%b1Ϊ���ز��1����Ԫ�����
b2=sigmoid(alpha_2-gamma_2);%b2Ϊ���ز��2����Ԫ�����

beta=omega_1*b1+omega_2*b2;%betaΪ�������Ԫ������
y_c=sigmoid(beta-theta);%y_cΪ�������Ԫ�����

Y(j)=y_c;%����j����Ʒ����MLP�����ֵ��¼��Y(j)��

%����Ϊ����BP�㷨���¸�����ֵ
g=y_c*(1-y_c)*(y(j)-y_c);%gΪcost function���������Ԫ������beta�ĵ������෴��

e1=b1*(1-b1)*omega_1*g;%e1Ϊcost function�����ز��1����Ԫ������alpha_1�ĵ������෴��
e2=b2*(1-b2)*omega_1*g;%e2Ϊcost function�����ز��2����Ԫ������alpha_2�ĵ������෴��

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
%���ϸ���Ϊ������ʽ������²���ֵ

E=E+0.5*(y_c-y(j)).^2;%����cost function

end

if E<0.0001 %cost function<0.0001,��ֹͣѵ��������������θ���17����Ʒ���²���ֵ
    break;
end

E=0; %������Ʒ����һ��֮��E��0

end


%����sigmoid����
function output = sigmoid(x)
output =1./(1+exp(-x));
end







