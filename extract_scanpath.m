function scanpath = extract_scanpath(EyeData_Example,task,n_observer,istim,K)
%Inputs: 
%EyeData_Example is a structrure. 1st level: experimental conditions. 
%2nd level: x_pos and y_pos, 3D matrix: observer x stimuli x time
%task: string containing current experimental condition name
%n_observer: current observer number(s)
%istim: current stimulus number
%K: number of HMM states (to discard scanpath with less fixations than
%state)

%Output:
%scanpath: cell containing the scanpath of  n_observer on istim 
%(time x coordinates)


scanpath=cell(1);

for isub= n_observer
    if size(EyeData_Example.(task).x_pos,1)>=isub
        x_pos=squeeze(EyeData_Example.(task).x_pos(isub,istim,:));
        y_pos=squeeze(EyeData_Example.(task).y_pos(isub,istim,:));
        
        x_nan=isnan(x_pos);
        y_nan=isnan(y_pos);
        xy_nan=x_nan|y_nan;
        y_pos(xy_nan)=[];
        x_pos(xy_nan)=[];
       
        if length(x_pos)<max(K)+1 % If less fixation than number of state, continue
            continue
        end
        if length(unique([x_pos y_pos]))==2
            continue
        end
        
        if length(n_observer)==1 % To extract one scanpath
            scanpath{1,1}=[x_pos y_pos];
        else
            scanpath{isub,1}=[x_pos y_pos]; % To extract a group of scanpaths
        end
    end
end