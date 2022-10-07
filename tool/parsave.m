function  [outputArg1] =parsave(dirobj,filename,var1)

save([dirobj filename '.mat'],'var1','-v7.3');
%save([dirobj filename '.png'],'var1');

outputArg1=1;