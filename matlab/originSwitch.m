function [x, y] = originSwitch(x ,y, width, height) 
% ORIGINSWITCH2 transforms x,y from cartesian coordinates to the coordinate
% system of another image
x = x + repmat(width/2,length(x),1);
y = -y + repmat(height/2,length(y),1);
end