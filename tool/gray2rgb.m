function rgbimg = gray2rgb(img)
[rows,cols]=size(img);
 r=zeros(rows,cols);
 g=zeros(rows,cols);
 b=zeros(rows,cols);
 r=double(img);
 g=double(img);
 b=double(img);
 rgbimg=cat(3,r,g,b);
end