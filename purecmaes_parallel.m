function [xmin] = purecmaes_parallel(Fitnessfct,xmean,varMin,varMax)
% CMA-ES: Evolution Strategy with Covariance Matrix Adaptation for nonlinear function minimization.
% <The CMA evolution strategy: A tutorial> DOI:10.48550/arXiv.1604.00772;
% Fitnessfct,目标函数句柄
% xmean = rand(N,1),初始决策变量
% varMin,决策变量的下界
% varMax,决策变量的上界
% -------------------- 初始化 --------------------------------
% 用户定义参数

N=length(xmean);          % 问题维度
xmean=reshape(xmean,[N,1]);% 调整决策变量及其上下界为列向量
varMin=reshape(varMin,[N,1]);
varMax=reshape(varMax,[N,1]);

sigma = 0.5;              % 初始步长
stopfitness = 0;          % 停止条件：适应度小于此值
stopeval = 1e3*N^2;       % 停止条件：最大评估次数

% 策略参数设置：选择
lambda = 4 + floor(3*log(N))*5;      % 种群大小
mu = lambda/2;                       % 父代数量
weights = log(mu+0.5) - log(1:mu)';  % 重组权重
mu = floor(mu);
weights = weights/sum(weights);       % 归一化权重
mueff = sum(weights)^2 / sum(weights.^2); % 有效选择质量

% 策略参数设置：适应
cc = (4 + mueff/N) / (N + 4 + 2*mueff/N);  % C的累积时间常数
cs = (mueff + 2) / (N + mueff + 5);        % σ的累积时间常数
c1 = 2 / ((N+1.3)^2 + mueff);              % 协方差矩阵秩1更新学习率
cmu = min(1-c1, 2*(mueff-2+1/mueff)/((N+2)^2 + 2*mueff/2)); % 秩μ更新学习率
damps = 1 + 2*max(0, sqrt((mueff-1)/(N+1))-1) + cs; % σ的阻尼系数

% 初始化动态参数
pc = zeros(N,1); ps = zeros(N,1);    % 进化路径
B = eye(N); D = eye(N);              % 协方差矩阵分解
C = B*D*(B*D)';                      % 初始协方差矩阵
eigeneval = 0;                       % 特征分解计数器
chiN = N^0.5*(1-1/(4*N)+1/(21*N^2)); % E||N(0,I)||的期望

%% -------------------- 主循环 --------------------------------
counteval = 0;g=1;
while counteval < stopeval
    % 生成并评估λ个子代
    arx = zeros(N, lambda);
    arz = zeros(N, lambda);
    arfitness = zeros(1, lambda);
    for k = 1:lambda
        arz(:,k) = randn(N,1);                % 采样标准正态分布
        xTemp = xmean + sigma*(B*D*arz(:,k)); % 生成候选解
        % 使用反射法处理超边界的变量
        xIndL=xTemp<varMin;xTemp(xIndL)=2*varMin(xIndL)-xTemp(xIndL);
        xIndU=xTemp>varMax;xTemp(xIndU)=2*varMax(xIndU)-xTemp(xIndU);
        arx(:,k)=xTemp;
        % 如果存在超边界的情况，则反过来更新arz
        if sum(xIndL+xIndU)>0
            arz(:,k)=(B*D)\((xTemp-xmean)/sigma);
        end
    end
    % 启用并行池来计算适应度函数，缩短计算时间
    parfor k=1:lambda
        arfitness(k) = Fitnessfct(arx(:,k)); % 评估适应度
    end
    counteval = counteval + lambda;

    % 按适应度排序并计算加权均值
    [arfitness, arindex] = sort(arfitness); % 最小化
    xmean = arx(:, arindex(1:mu)) * weights; % 更新均值
    zmean = arz(:, arindex(1:mu)) * weights;

    % 更新进化路径
    ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * (B*zmean);
    hsig = norm(ps)/sqrt(1-(1-cs)^(2*counteval/lambda))/chiN < 1.4 + 2/(N+1);
    pc = (1-cc)*pc + hsig*sqrt(cc*(2-cc)*mueff)*(B*D*zmean);

    % 调整协方差矩阵
    C = (1-c1-cmu)*C + c1*(pc*pc' + (1-hsig)*cc*(2-cc)*C) + ...
        cmu * (B*D*arz(:,arindex(1:mu))) * diag(weights) * (B*D*arz(:,arindex(1:mu)))';

    % 调整步长σ
    sigma = sigma * exp((cs/damps)*(norm(ps)/chiN - 1));

    % 更新协方差矩阵的分解B,D,C（每间隔若干代）
    if counteval - eigeneval > lambda/(c1+cmu)/N/10
        eigeneval = counteval;
        C = triu(C) + triu(C,1)'; % 强制对称
        [B,D] = eig(C);           % 特征分解
        D = diag(sqrt(diag(D)));  % D为标准差矩阵
    end

    % 输出本次迭代的最优适应度值
    disp(['Iteration : ' num2str(g) '; Best Cost = ' num2str(arfitness(1))]);
    g=g+1;

    % 检查停止条件
    if arfitness(1) <= stopfitness
        break;
    end

    % 处理平适应度问题
    if arfitness(1) == arfitness(ceil(0.7*lambda))
        sigma = sigma * exp(0.2 + cs/damps);
        disp('警告：适应度无变化，建议检查目标函数');
    end
end

% 返回最佳解
xmin = arx(:, arindex(1));
disp(['总评估次数: ' num2str(counteval) ', 最佳适应度: ' num2str(arfitness(1))]);
end
