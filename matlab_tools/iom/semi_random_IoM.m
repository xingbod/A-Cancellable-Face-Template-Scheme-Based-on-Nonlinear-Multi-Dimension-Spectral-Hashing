function model = semi_random_IoM(data, opts)
X = data.X;
S = data.S;

K = opts.K;
L = opts.L*2;
lambda = opts.lambda;
beta = opts.beta;

[dx, N] = size(X);
S = double(S);

if opts.S==1
    S = double(S);
elseif opts.S==2
    S = eye(N);
elseif opts.S==3
    S = ones(N);
end


% training
bSize = min(500, N); % batch size

Px = cell(1, L);
alpha = ones(1, L)/L;
F = lambda-(lambda+1)*S;
A = ones(N); % Initial weights of every pair are 1
totalTime = 0;
DEBUG = 0;
sizePos = sum(S(:));
Npairs = lambda*(N*N-sizePos)+sizePos;
nb = 5;


eta = 4000.0; % learing rate
momentum = 0.0; % weight update momentum
err = zeros(L+1, 1);
err(1)=1;
for n = 1 : L
    fprintf('Learn bit %d.\n', n); tStart = tic;
    if opts.gaussian
        Wx = normc((mvnrnd(zeros(dx, 1), diag(ones(dx, 1)), K))');
    else
        Wx=laprnd(dx, K, 0, 1);
    end
    D = F.*A;
    Np = (sum(S(:).*A(:)))/Npairs;
    
        if mod(n, 10) == 0
            A = ones(N);
        end
    
    % profile initialization error
    
    
    dWx = 0;
    
    bid = randperm(N, bSize);
    Xb = X(:, bid);
    Zxb = Wx'*Xb;
    Xeb = exp(beta*bsxfun(@minus, Zxb, max(Zxb)));
    Hxb = bsxfun(@rdivide, Xeb, sum(Xeb));
    
    Db = D(bid, bid);
    
    Hsxb = Hxb*Db;
    dWx_old = dWx;
    dWx = Xb*((Hxb.*Hsxb)'-bsxfun(@times, diag(Hsxb'*Hxb), Hxb'))/(bSize*bSize);
    dWx = eta*dWx + momentum*dWx_old;
    Wx = Wx - dWx;
    
    
    Zx = Wx'*X;
    Xe = exp(beta*bsxfun(@minus, Zx, max(Zx)));
    Hx = bsxfun(@rdivide, Xe, sum(Xe));
    % profile initialization error
    err(n+1) = trace(Hx*D*Hx')/Npairs+Np;
    
        if err(n+1) > err(n)
            eta = eta * 0.9;
        else
            eta = eta * 1.05;
        end
    
    eps_0 = err(n);
    eps_e =  err(n+1);
    % update weights of training pairs eps The epsilon of the machine (short: eps) is the minimum distance that a floating point arithmetic program like Matlab can recognize between two numbers x and y
    alpha(n) = log((1-eps_e)/(eps_e+eps));
    Sc = Hx'*Hx;
    A = A.*exp(alpha(n)*(abs(Sc-S).^1));
    A = A*N*N/sum(A(:));
    
    tElapsed = toc(tStart);
    totalTime = totalTime + tElapsed;
    fprintf('%d iterations, %.5f seconds, error %.4f->%.4f\n', n, tElapsed, eps_0, eps_e);
    Px{n} = Wx;
end

[B,I] = maxk(alpha,opts.L)
model.Wx = cell2mat(Px(I));
model.weight =alpha(I);

avTime = totalTime / L;
fprintf('Training complete. Average time: %.5f seconds per bit.\n', avTime);


