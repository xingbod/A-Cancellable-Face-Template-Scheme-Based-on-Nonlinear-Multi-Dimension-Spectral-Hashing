function model = learning_IoM(data, opts)
X = data.X;
S = data.S;

K = opts.K;
L = opts.L;
lambda = opts.lambda;
beta = opts.beta;

S = double(S);
[dx, N] = size(X);

% training
maxIters = 201;
bSize = min(500, N); % batch size
delta = 1e-10; % stopping condition
momentum = 0.0; % weight update momentum

mu = 0.0;
Px = cell(1, L);
Py = cell(1, L);
alpha = ones(1, L)/L;
F = lambda-(lambda+1)*S;
A = ones(N); % Initial weights of every pair are 1
totalTime = 0;
DEBUG = 0;
sizePos = sum(S(:));
Npairs = lambda*(N*N-sizePos)+sizePos;
nb = 5;
for n = 1 : L
    fprintf('Learn bit %d.\n', n); tStart = tic;
    if opts.gaussian
        Wx = normc((mvnrnd(zeros(dx, 1), diag(ones(dx, 1)), K))');
    else
        Wx=laprnd(dx, K, 0, 1);
    end
    
    D = F.*A;
    Np = (sum(S(:).*A(:)))/Npairs;
    
    if mod(n, 4) == 0
        A = ones(N);
    end
    
    err = zeros(maxIters+1, 1);
    gnorm = zeros(maxIters, 1);
    evar = zeros(maxIters, 1);
    rate = zeros(maxIters, 1);
    
    % profile initialization error
    Xe = exp(beta*bsxfun(@minus, Wx'*X, max(Wx'*X)));
    Hx = bsxfun(@rdivide, Xe, sum(Xe));
    err(1) = trace(Hx*D*Hx')/Npairs+Np;
    
    dWx = 0;
    eta = 4000.0; % learing rate
    for i = 2 : maxIters
        % randomly sample a batch
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
        
        gnorm(i-1) = mean(abs(dWx(:)));
        if mod(i-1, nb) == 0
            Zx = Wx'*X;
            Xe = exp(beta*bsxfun(@minus, Zx, max(Zx)));
            Hx = bsxfun(@rdivide, Xe, sum(Xe));
            epoc = (i-1)/nb+1;
            % profile initialization error
            err(epoc) = trace(Hx*D*Hx')/Npairs+Np;
            if epoc >= 2
                % change learning rate adaptively
                if err(epoc) > err(epoc-1); eta = eta * 0.5;
                else eta = eta * 1.05; end
                evar(epoc-1) = abs(err(epoc-1)-err(epoc))/(max(err(epoc-1), err(epoc))+eps);
                rate(epoc-1) = eta;
                if DEBUG; fprintf('Iter %d, f(x) = %f, error_var = %f\n', i-1, err(epoc), evar(epoc-1)); end
                % check stopping condition
                if abs(evar(epoc-1)) <= delta; end
            end
        end
    end
    eps_0 = err(1);
    eps_e = err(epoc);
    % update weights of training pairs
    alpha(n) = log((1-eps_e)/(eps_e+eps));
    Sc = Hx'*Hx;
    A = A.*exp(alpha(n)*(abs(Sc-S).^1));
    A = A*N*N/sum(A(:));
    tElapsed = toc(tStart);
    totalTime = totalTime + tElapsed;
    fprintf('%d iterations, %.5f seconds, error %.4f->%.4f\n', i-1, tElapsed, eps_0, eps_e);
    Px{n} = Wx;
end
model.Wx = cell2mat(Px);

avTime = totalTime / L;
fprintf('Training complete. Average time: %.5f seconds per bit.\n', avTime);