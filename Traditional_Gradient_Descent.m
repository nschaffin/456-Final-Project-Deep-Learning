function Traditional_Gradient_Descent
    % Data ----------------------------------------------------------------
    x1 = [0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7];
    x2 = [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6];
    y = [ones(1,5) zeros(1,5); zeros(1,5) ones(1,5)];


    % Init Weights and Biases ---------------------------------------------
    rng(5000);
    W2 = 0.5*randn(2,2); W3 = 0.5*randn(3,2); W4 = 0.5*randn(2,3);
    b2 = 0.5*randn(2,1); b3 = 0.5*randn(3,1); b4 = 0.5*randn(2,1);
    
    % Foward and Backward Propogate ---------------------------------------
    eta = 0.05; % learning rate
    Niter = 1e6; % number of SG iterations
    savecost = zeros(Niter,1); % value of cost function at each iteration
    for counter = 1:Niter
        W2_delta_avg = zeros(size(W2,1),1);
        W3_delta_avg = zeros(size(W3,1),1);
        W4_delta_avg = zeros(size(W4,1),1);
        b2_delta_avg = zeros(length(b2),1);
        b3_delta_avg = zeros(length(b3),1);
        b4_delta_avg = zeros(length(b4),1);
        for k=1:length(x1)
            x = [x1(k); x2(k)];
            % Forward pass
            a2 = sigmoid(x,W2,b2);
            a3 = sigmoid(a2,W3,b3);
            a4 = sigmoid(a3,W4,b4);
            % Backward pass
            delta4 = a4.*(1-a4).*(a4-y(:,k));
            delta3 = a3.*(1-a3).*(W4'*delta4);
            delta2 = a2.*(1-a2).*(W3'*delta3);

            W2_delta_avg = W2_delta_avg + delta2 * x';
            W3_delta_avg = W3_delta_avg + delta3 * a2';
            W4_delta_avg = W4_delta_avg + delta4 * a3';

            b2_delta_avg = b2_delta_avg + delta2;
            b3_delta_avg = b3_delta_avg + delta3;
            b4_delta_avg = b4_delta_avg + delta4;
        end

        % Average the gradient
        W2_delta_avg = mean(W2_delta_avg);
        W3_delta_avg = mean(W3_delta_avg);
        W4_delta_avg = mean(W4_delta_avg);
        b2_delta_avg = mean(b2_delta_avg);
        b3_delta_avg = mean(b3_delta_avg);
        b4_delta_avg = mean(b4_delta_avg);

        % Gradient step
        W2 = W2 - eta*W2_delta_avg;
        W3 = W3 - eta*W3_delta_avg;
        W4 = W4 - eta*W4_delta_avg;
        b2 = b2 - eta*b2_delta_avg;
        b3 = b3 - eta*b3_delta_avg;
        b4 = b4 - eta*b4_delta_avg;
        % Monitor progress
        newcost = cost(W2,W3,W4,b2,b3,b4) % display cost to screen
        savecost(counter) = newcost;
    end

    % Show decay of cost function
    save costvec
    semilogy([1:1e4:Niter],savecost(1:1e4:Niter))

    function costval = cost(W2,W3,W4,b2,b3,b4)
        costvec = zeros(10,1);
        for i = 1:10
            x =[x1(i);x2(i)];
            a2 = sigmoid(x,W2,b2);
            a3 = sigmoid(a2,W3,b3);
            a4 = sigmoid(a3,W4,b4);
            costvec(i) = norm(y(:,i) - a4,2);
        end
        costval = norm(costvec,2)^2;

    end % of nested function
end
