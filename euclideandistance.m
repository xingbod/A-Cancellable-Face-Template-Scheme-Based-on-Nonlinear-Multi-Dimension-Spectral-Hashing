function [distance] = euclideandistance(facenet_gallery,facenet_probe_o1)
%EUCLIDEANDISTANCE Summary of this function goes here
%   Detailed explanation goes herescore_o1 =1./(pdist2( facenet_gallery,facenet_probe_o1,  'euclidean')+0.001);
for i=1: size(facenet_gallery,1)
    for j=1: size(facenet_probe_o1,1)
        distance(i,j)=norm(facenet_gallery(i,:)-facenet_probe_o1(j,:))/(norm(facenet_gallery(i,:))+norm(facenet_probe_o1(j,:)));
    end
end
end

