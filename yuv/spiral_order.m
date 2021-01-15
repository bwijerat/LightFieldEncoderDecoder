function [img_list] = spiral_order(a, m, n)

img_list = [];

%     Setting Boundaries for the given matrix

    t = 1;
    b = m;
    l = 1;
    r = n;
    dr = 1;

%     t: top of the matrix
%     b: botom of the matrix
%     l: left of the matrix
%     r: Right of yhe matrix
%     dr: printing direction
%     m: Number of rows in the matrix
%     n: Number of Columns in the matrix

    while (t <= (b+1) && l <= (r+1))
        
        fprintf('.')
        if dr == 1
            for i  = l : r
                img_list = [img_list, a(t, i, :, :, :)];
            end
            t = t + 1;
            dr = 2;
            
        elseif dr == 2
            for i = t : b
                img_list = [img_list, a(i, r, :, :, :)];
            end
            r = r - 1;
            dr = 3;
            
        elseif dr == 3
            for i = r : -1 : l
                img_list = [img_list, a(b, i, :, :, :)];
            end
            b = b - 1;
            dr = 4;
            
        elseif dr == 4
            for i = b : -1 : t
                img_list = [img_list, a(i, l, :, :, :)];
            end
            l = l + 1;
            dr = 1;
            
        end
        
    end
end