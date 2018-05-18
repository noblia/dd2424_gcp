function [offset1, offset2] = getOffset(coordinate, size, border)
offset = round((size-1)/2);

comp = min(coordinate, border - coordinate);

if comp <= offset
    offset1 = offset - abs(offset-comp)-1;
    offset2 = offset*2 - offset1;
    if coordinate > border - coordinate
        % can't subract big number if the coordinat is on the max-border
        temp = offset1;
        offset1 = offset2;
        offset2 = temp;
    end
else
    offset1 = offset; offset2 = offset;
end
