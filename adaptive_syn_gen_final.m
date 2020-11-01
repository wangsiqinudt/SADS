function [new_outlier,edgeset,new_target] = adaptive_syn_gen_final(target, ed_t, coef)
dist = squareform(pdist(+target, 'euclidean'));     

% select edge and interior patterns
k = round(5*log10(size(target,1)));
[l,nn,knn_dis, target_dis,~] = epd(+target, dist, k, 'train');
edge_idx = (l>=1-ed_t);
repel_dis = mean(knn_dis(edge_idx));
% knn_mean_dis = mean(knn_dis);
new_outlier = (+target(edge_idx,:))+repmat(coef*repel_dis,[sum(edge_idx),size(target,2)]).*nn(edge_idx,:);
edgeset = +target(edge_idx,:);
% interiorset = +target(int_idx,:);
new_target = (+target)-repmat(target_dis,[1,size(target,2)]).*nn;


% pruning
% [~,s_outlier] = epd(+target, [new_outlier;new_outlier2], 0.75*knn_mean_dis, 'test');
end

function [l, nn, knn_dis, target_dis, idx_mat] = epd(data, dist_or_test, k, mode)
if(strcmp(mode,'train'))
% l: the percentage of vectors on the normal vector side
% nn: the normalized normal vector
% un: unnormalized normal vector
% knn_dis: the average dis. from the point to its knn
% idx_mat: the index of sorted neighbor, a n*(n-1) matrix
    n = size(dist_or_test,1);
    l = zeros(size(data,1),1);
    target_dis = zeros(size(data,1),1);% the distance to repel an edge pattern into the target set as a new target datum
    knn_dis = zeros(size(data,1),1);
    nn = zeros(size(data,1),size(data,2));
    tmp = diag(-ones(n,1));
    dist_or_test = dist_or_test+tmp;
    [~,idx_mat] = sort(dist_or_test,2);
    % remove the point itself from its neighbors
    idx_mat = idx_mat(:,2:end);
    for i=1:n 
        k_idx = idx_mat(i,1:k);
        tmp = dist_or_test(i,k_idx(1:k));
        knn_dis(i) = mean(tmp);
        v = repmat(data(i,:),[k,1])-data(k_idx,:);
        vv = sum(v.*repmat(1./sqrt(sum(v.^2,2)+1e-10),[1,size(data,2)]));
        ll = vv*v';
        nn(i,:) = vv./(sqrt(sum(vv.^2))+1e-10);
        ind = (ll>=0);ind2 = (ll<0);
        l(i) = sum(ind)/k;
        % calculate the target repel distance: the minimal distance to the
        % same side of normal vector in knn
        ll(ind2) = inf;[~,ind] = min(ll);
        target_dis(i) = nn(i,:)*(v(ind,:))';
    end    
else
% pruning mode
    test = dist_or_test;
    thr = k^2;
    XXh1 = sum(data.^2,2)*ones(1,size(test,1));
    XXh2 = sum(test.^2,2)*ones(1,size(data,1));
    dist = XXh1+XXh2' - 2*data*test';
    min_dist = min(dist,[],1);
    l = find(min_dist>thr);
    nn = test(l,:);
end
end
