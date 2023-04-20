
rng(5000);
W2 = reshape(0.5*randn(2,2),[4,1]); 
W3 = reshape(0.5*randn(3,2),[6,1]); 
W4 = reshape(0.5*randn(2,3),[6,1]);
b2 = 0.5*randn(2,1); 
b3 = 0.5*randn(3,1); 
b4 = 0.5*randn(2,1);

p = [W2; W3; W4; b2; b3; b4];
x = lsqnonlin(@Cost, p, lb = ones(23,1) * -10, ub = ones(23,1) * 10);
exitflag
d = linspace(0,1);
y = linspace(0,1);
%plot(d,y,'ko',d,)



function cost = Cost(p)
    x1 = [0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7];
    x2 = [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6];
    y = [ones(1,5) zeros(1,5); zeros(1,5) ones(1,5)];
    cost = 0;

    weights = {reshape(p(1:4), [2,2]), reshape(p(5:10),[3,2]), reshape(p(11:16), [2,3])};
    biases = {reshape(p(17:18), [2,1]), reshape(p(19:21), [3,1]), reshape(p(22:end), [2,1])};
    for i=1:10
        cost = cost + (y(:,i) - F([x1; x2], weights, biases));
    end
    cost = 1/20 * cost;
end

function y_approx = F(x, weights, biases)
    a = x;
    for i=1:3
        a = sigmoid(a, weights{i}, biases{i});
    end
    y_approx = a;
end