function [w1,best_para] = ocsvm_validation_with_target(x, xte, w, fracrej, range, varargin)


% Check some things:
if ~isa(w,'char')
	error('Expecting the name (string!) of the classifier');
end
if length(fracrej)>1
	error('Fracrej should be a scalar');
end


% AI!---------------^

% OCCS having K hyperparameters
K = length(range);

cur_pos = ones(1,K);
para = zeros(1,K);
max_times = 1;
for i=1:K
    % The total number of parameters combination.
    max_times = max_times*length(range{i});
end


cur_err = 2;
for j=1:K
    best_para(j) = range{j}(cur_pos(j));
end
for t=1:max_times
    
    % Get the current para combination.
    for j=1:K
        para(j) = range{j}(cur_pos(j));
    end
    

    ww1 = feval(w, x, fracrej, para, varargin{:});
    res = dd_error(xte,ww1);
    fn = res(1);
    fp = res(2);
       
    fracout = 0.5*fn+0.5*fp;
    if(fracout<cur_err)
        best_para = para;
        cur_err = fracout;
        w1 = ww1;
    end
    
    % Update the index of hyperparameters.
    cur_pos(1) = cur_pos(1)+1;
    for j=1:K
        if cur_pos(j)>length(range{j})
            cur_pos(j) = 1;
            if j~= K
                cur_pos(j+1) = cur_pos(j+1)+1;
            end
        else
            break;
        end
    end
end


% w1 = feval(w, x, fracrej, best_para, varargin{:});

return