function  [outputArg1] =parsave2(dirobj,filename,L,W)

save([dirobj filename '.mat'],'L','W','-v7.3');
%save([dirobj filename 'eq3.mat'],'var2','-v7.3');

outputArg1=1;