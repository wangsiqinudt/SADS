%CHECKTOOLBOX Check avialability of toolbox
%
%     N = CHECKTOOLBOX(TOOLBOX)
%
% INPUT
%		TOOLBOX	 String with toolbox name
%
% OUTPUT
%		N        Flag, 1/0 if the toolbox is/isn't in the path
%
% DESCRIPTION
% Checks whether TOOLBOX is in the path.

function n = checktoolbox(name)

  if nargin == 0 || ~ischar(name)
    error('No input string found')
  end

  if nargout == 0

    switch lower(name)
      case('stats_svmtrain')
        if exist('stats/svmtrain') ~= 2
          error([newline 'Matlab''s Support Vector Classifier, svmtrain, ' ...
            newline 'is not found. Add the stats toolbox to the path or upgrade it.'])
        end
      case('naivebayes')
        if exist('NaiveBayes\fit') ~= 2
          error([newline 'Matlab''s naive Bayes classifier, NaiveBayes, ' ...
            newline 'is not found. Add the stats toolbox to the path or upgrade it.'])
        end
      case('classificationknn')
        if exist('ClassificationKNN.m') ~= 2
          error([newline 'Matlab''s nearest neighbor classifier, ClassificationKNN, ' ...
            newline 'is not found. Add the stats toolbox to the path or upgrade it.'])
        end
      case('classificationdiscriminant')
        if exist('ClassificationDiscriminant') ~= 2
          error([newline 'Matlab''s linear classifier, ClassificationDiscriminant, ' ...
            newline 'is not found. Add the stats toolbox to the path or upgrade it.'])
        end
      case('classificationtree')
        if exist('ClassificationTree.m') ~= 2
          error([newline 'Matlab''s decision tree classifier, ClassificationTree, ' ...
            newline 'is not found. Add the stats toolbox to the path or upgrade it.'])
        end
      case('libsvm')
        if exist('libsvm/svmtrain') ~= 3
          error([newline 'The LIBSVM package is not found.' newline ...
          'Add it to the Matlab path or download it from ' ...
          '<a href="http://www.csie.ntu.edu.tw/~cjlin/libsvm/">here</a>.'])
        end
      case('diplib')
        if exist('diplib','dir') ~= 7
          error([newline 'The DIPIMAGE package is needed and not found.' ...
          newline 'Add it to the Matlab path or download it from ' ...
          '<a href="http://www.diplib.org/">here</a>.'])
        end
      otherwise
        if exist(name,'dir') ~= 7
          error([newline 'The ' upper(name) ' toolbox is needed. ' ...
            'Please add it to the path.'])
        end
    end

  else

    n = exist(name,'dir') == 7;

  end

return
  
  
    
      