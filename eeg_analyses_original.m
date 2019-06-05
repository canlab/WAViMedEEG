%remove artifact
%three data sets per file (1-leading up to thump but no thump period.
%2-distance between thumps after thumps, 3-everything else....)

eeg_fnames=filenames(fullfile('~/Desktop/PainStudyFiles/*SMS.eeg'));
art_fnames=filenames(fullfile('~/Desktop/PainStudyFiles/*SMS.art'));
evt_fnames=filenames(fullfile('~/Desktop/PainStudyFiles/*SMS.evt'));

%load in all the raw eeg data
for sub=1:length(eeg_fnames)
   data_temp=load(eeg_fnames{sub});
  
   data{sub}=data_temp;
end

%load in the evt files to make sure we're presenting 40 stims per person.
%The number of stimuli presented range from 79 to 82. There is also some
%variance in the total number of samples collected.
for sub=1:length(evt_fnames)
   evt{sub}=load(evt_fnames{sub});
end
for sub=1:29
    number_stimuli(sub,1)=size(find(evt{sub}(:)>0),1);
    number_stimuli(sub,2)=size(data{sub},1);
end

%let's remove some artifacts. load in art files first, find indices where
%artifacts occur (regardless of artifact amplitude), set values in data
%variable (raw eeg data) equal to NaN if an artifact occurs. Doing this to
%maintain the original length of the collection.
for sub=1:length(art_fnames)
    art{sub}=load(art_fnames{sub});
    art_index=find(art{sub}>0);
    data{sub}(art_index)=NaN;
end

%Now we'll create the three different data sets described above.(1-leading up to thump but no thump period.
%2-distance between thumps after thumps, 3-everything else....)







