%parpool(24);
eq1_times = 100;                %eq1缩放倍数
eq2_times = 100;                %eq2缩放倍数
eq3_times = 0.8;                 %eq3缩放倍数
eq3m_times = 0.8;
gama = 1;                       %γtimes for Z(eq2)
thr = 0.8;
img_path='/home/ubuntu/JP/data/VOC2012/pascal_2012_scribble/';
scr_path='/home/ubuntu/JP/data/VOC2012/pascal_2012_scribble_scribble/'; 	%scribble path
scr_path_list ='/home/ubuntu/JP/data/VOC2012/pascal_2012_scribble_scribble/*.png';
sav_path = '/home/ubuntu/JP/data/VOC2012/spiex/';		%save_path,
sav_path2 = '/home/ubuntu/JP/data/VOC2012/Laplace/';
mkdir(sav_path);
list=dir(scr_path_list);
list_mat = dir(sav_path2);
listcell = struct2cell(list_mat);

start_time=cputime;


parfor i=1:length(list)
    if_saved = find(strcmp(listcell,[list(i).name(1:end-4) '.mat']));
    if  ~isempty(if_saved)
        %disp('skip')
        continue
    end
    disp(i)
    
    img=imread([img_path list(i).name(1:end-4) '.jpg']);
    scr=imread([scr_path list(i).name(1:end-4) '.png']);
    
    %超像素分割
    [L,N]=superpixels(img,800);
    if_save = parsave(sav_path,[list(i).name(1:end-4)],L);
    %continue
    
    %计算图片梯度
    img_size = size(img);
    if numel(img_size)<=2
        img = gray2rgb(img);
    end
    img_gray=double(rgb2gray(img));
    [img_grad1,img_grad2]=gradient(img_gray);  %获取梯度
    img_grad=cat(3,img_grad1,img_grad2);
    grad_min=min(img_grad(:));
    grad_max=max(img_grad(:));
    
    %计算邻接关系
    glcms = graycomatrix(L,'NumLevels',N,'GrayLimits',[1,N],'Offset',[0,1;1,0]);
    glcms = sum(glcms,3);    % add together the two matrices
    glcms = glcms + glcms.'; % add upper and lower triangles together, make it symmetric
    glcms(1:N+1:end) = 0;    % set the diagonal to zero, we don't want to see "1 is neighbor of 1"
    [I,J] = find(glcms);     % returns coordinates of non-zero elements
    neighbors = [J,I];
    
    %预设变量
    eq1=ones(N,21)*inf;
    eq1_1=ones(1,21)*inf;
    temp=unique(scr);
    temp=temp(temp~=255);
    eq1_1(sub2ind(size(eq1_1),ones(size(temp)),temp+1))=-log(1/length(temp));%
    eq3_c=zeros(N,25*3);
    eq3_t=zeros(N,10*2);
    eq3=zeros(N);
    for k=1:N
        %% 计算公式 2
        temp=unique(scr(L==k));
        temp=temp(temp~=255);
        eq1(sub2ind(size(eq1),k*ones(size(temp)),temp+1))=0;
        if isempty(temp)
            eq1(k,:)=eq1_1.';
        end
        %% 计算公式 4 中直方图特征
        %计算颜色空间直方图
        hist_c=[];
        for c=1:3
            img_c=img(:,:,c);
            hist=histogram(img_c(L==k));
            hist.BinEdges=[0,255];hist.NumBins=25;
            hist_c=[hist_c hist.Values/length(img_c(L==k))];
        end
        eq3_c(k,:)=hist_c;
        %计算纹理直方图
        hist_t=[];
        for t=1:2
            img_t=img_grad(:,:,t);
            hist=histogram(img_t(L==k));
            hist.BinEdges=[grad_min,grad_max];hist.NumBins=10;
            hist_t=[hist_t hist.Values/length(img_t(L==k))];
        end
        eq3_t(k,:)=hist_t;
    end
    %% 根据邻接关系计算公式 4
    for n=1:length(I)
        if eq3(neighbors(n,1),neighbors(n,2))==0 && eq3(neighbors(n,2),neighbors(n,1))==0
            temp=norm(eq3_c(neighbors(n,1),:)-eq3_c(neighbors(n,2),:))/25+norm(eq3_t(neighbors(n,1),:)-eq3_t(neighbors(n,2),:))/100;
            eq3(neighbors(n,1),neighbors(n,2))=exp(temp);
        end
    end
    eq3=eq3+eq3';
    
    
    %% 保存eq1,eq3,以及superpixels的结果L matrices
    eq3 = log(eq3);
    eq3 = exp(-eq3*5);
    eq3(eq3 == Inf) = 0;
    count = N;
    eq3m = min(min(eq3(eq3 > 0)));
    eq3 = eq3 - eq3m*eq3m_times;
    eq3(eq3<0) = 0;
    eq3 = normalize(eq3,'range');
    eq3 = eq3*eq3_times;
    
    La = sum(eq3);
    La = diag(La);
    La = La - eq3;
    
    if_save = parsave2(sav_path2,[list(i).name(1:end-4)],La,eq3);
end
end_time=cputime;
op_time=end_time-start_time;
