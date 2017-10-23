function [data, sid_names, sid_trials] = read_xls_fixations(xlsname, opt)
% read_xls_fixations - read an EXCEL file with fixation data
%
%  [data, subject_names, trial_names] = read_xls_fixations(xlsname, opt)
%
% Expected header cells in the spreadsheet:
%   SubjectID = subject ID
%   TrialID   = trial ID for subject
%   FixX      = fixation X-location
%   FixY      = fixation Y-location
%   FixD      = fixation duration in milliseconds (optional)
%
% SubjectID and TrialID can be either strings or numbers.
% FixX and FixY must be numbers.
% FixD is a number (milliseconds).
%
% Data will be separated by SubjectID and TrialID. 
% For each trial, fixations are assumed to be in sequential order.
%
% INPUT
%   xlsname - filename for the Excel spreedsheet (xls)
%   opt     - options (not used)
%
% OUTPUT
%   data - data cell array
%     data{i}         = i-th subject
%     data{i}{j}      = ... j-th trial
%     data{i}{j}(t,:) = ... [x y] location of t-th fixation 
%                        or [x y d] of t-th fixation (location & duration)
%
%   The subject/trials will be assigned numbers according to their order in the spreadsheet.
%   the following two outputs contain the original subject names and trial IDs:
%     subject_names{i} - the subject ID in the spreadsheet for the i-th subject
%     trial_names{i}{j} - the j-th trial ID for i-th subject
%
% ---
% Eye-Movement analysis with HMMs (emhmm-toolbox)
% Copyright (c) 2017-01-13
% Antoni B. Chan, Janet H. Hsiao, Tim Chuk
% City University of Hong Kong, University of Hong Kong

% VERSIONS:
%  2017-01-18 - added duration field FixD
%  2017-01-20 - issue an error if FixX, FixY, FixD values are strings.

fprintf('Reading %s\n', xlsname);

% read the XLS file
[ndata, tdata, rdata] = xlsread(xlsname);


% get the headers
headers = {rdata{1,:}};

% find the header indices
SID = find(strcmp('SubjectID', headers));
TID = find(strcmp('TrialID', headers));
FX  = find(strcmp('FixX', headers));
FY  = find(strcmp('FixY', headers));
FD  = find(strcmp('FixD', headers));
if length(SID) ~= 1
  error('error with SubjectID');
end
fprintf('- found SubjectID in column %d\n', SID);
if length(TID) ~= 1
  error('error with TrialID');
end
fprintf('- found TrialID in column %d\n', TID);
if length(FX) ~= 1
  error('error with FixX');
end
fprintf('- found FixX in column %d\n', FX);
if length(FY) ~= 1
  error('error with FixY');
end
fprintf('- found FixY in column %d\n', FY);
if length(FD) == 1
  fprintf('- found FixD in column %d\n', FD);
elseif length(FD) > 1
  error('error with FixD -- to many columns');
end

% initialize names and trial names
sid_names = {};
sid_trials = {};
data = {};

% read data
for i=2:size(rdata,1)
  mysid = rdata{i,SID};
  mytid = rdata{i,TID};
  
  if isstr(rdata{i,FX})
    error('Value for FixX is text, not a number.');
  end  
  if isstr(rdata{i,FY})
    error('Value for FixY is text, not a number.');
  end
  
  myfxy  = [rdata{i,FX}, rdata{i,FY}];
  if length(FD) == 1
    % include duration if available    
    if isstr(rdata{i,FD})
      error('Value for FixD is text, not a number.');
    end
    myfxy = [myfxy, rdata{i,FD}];
  end
  
  if isreal(mysid)
    mysid = sprintf('%g', mysid);
  end
  if isreal(mytid)
    mytid = sprintf('%g', mytid);
  end
  
  % find subject
  s = find(strcmp(mysid, sid_names));
  if isempty(s)
    % new subject
    sid_names{end+1,1} = mysid;
    sid_trials{end+1,1} = {};
    s = length(sid_names);
    data{s,1} = {};
  end
  
  % find trial
  t = find(strcmp(mytid, sid_trials{s}));
  if isempty(t)
    sid_trials{s,1}{end+1,1} = mytid;
    t = length(sid_trials{s});
    data{s,1}{t,1} = [];
  end
  
  % put fixation
  data{s,1}{t,1}(end+1,:) = myfxy;
end

fprintf('- found %d subjects:\n', length(sid_names));
fprintf('%s ', sid_names{:})
fprintf('\n');
for i=1:length(data)
  fprintf('    * subject %d had %d trials\n', i, length(data{i}));
end



