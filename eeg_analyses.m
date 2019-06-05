%The following code is intended for SMS thumper data


%load filenames of all SMS thumper data into 3 arrays, EEG, ART, and EVT
eeg_fnames=filenames(fullfile('*SMS.eeg'));
art_fnames=filenames(fullfile('*SMS.art'));
evt_fnames=filenames(fullfile('*SMS.evt'));

%spit error if length all 3 fnames inconsistent

n=length(eeg_fnames);

%%
%load in all the raw eeg data
for sub=1:n
    data{sub}=load(eeg_fnames{sub});
end

%%
%load in the evt files to make sure we're presenting 40 stims per person.
%The number of stimuli presented range from 79 to 82. There is also some
%variance in the total number of samples collected.
for sub=1:n
    evt{sub}=load(evt_fnames{sub});
end
for sub=1:n
    number_stimuli(sub,1)=size(find(evt{sub}(:)>0),1);
    number_stimuli(sub,2)=size(data{sub},1);
end

%%
%Removing artifacts
%load in art files first, find indices where artifacts occur (regardless of artifact amplitude)
%set values in data (raw eeg data) equal to 0 if an artifact occurs
% ^ maintains the original length of the collection.
for sub=1:n
    art{sub}=load(art_fnames{sub});
    art_index=find(art{sub}>0);
    %replace ART datapoints with zeros
    data{sub}(art_index)=0;
end

%%
%sanity check for consistent times between thumps
%vary, but almost always 0.6000 seconds (150 samples)
%0.5960 and 0.6040 appear often as well, so 1 sample error (?)
%also some 0s due to (?)
%sorry for the embedded loop :/
for sub=1:n
    numthumps=0;
    thump1=0;
    thump2=0;
    for sample=1:length(evt{sub})
        if evt{sub}(sample) == 1
            thump1=sample;
        end
        if evt{sub}(sample) == 2
            numthumps=numthumps+1;
            thump2=sample;
            thumplength=((thump2-thump1));
            thumps(numthumps,sub)=thumplength;
        end
    end
end
%%
%for every subject's SMS EEG file, split into 3 arrays
%1. eegStruct{sub}{1} = array of EEG data for 155 samples after each 1st thump
%each iteration should be a column = 155 samples
for sub=1:n
    onethump=find(evt{sub}==1)
    for thumpnum=1:length(onethump)
        try
            setOne{sub}{thumpnum}=data{sub}(onethump(thumpnum):onethump(thumpnum)+154,:)
        catch
            continue
        end
    end
end

%%
%2. eegStruct{sub}{2} = array of EEG data for 155 samples after each 2nd thump
%each iteration should be a column = 155 samples
for sub=1:n
    twothump=find(evt{sub}==2)
    for thumpnum=1:length(twothump)
        try
            setTwo{sub}{thumpnum}=data{sub}(twothump(thumpnum):twothump(thumpnum)+154,:)
        catch
            continue
        end
    end
end
%what if we pulled the two sets of indices for where thump1 and thump2 are
%but then pull like 155 samples for every subject? The advantage being we
%don't want to lose the second thump in some cases? Like there's 151
%samples in between thumps sometimes right?
%%
%3. eegStruct{sub}{3} = array of EEG data until the next 1st thump
%each iteration should be a column of data, varying length
% for sub=1:29
%     for thumpnum=1:length(twothump)
%         try
%                   
%         catch
% 
%         end
%     end
% end

 %%
% Listing out subject data types and counting
% 0 = control, 1 = pain, 2 = relief/follow-up
ncontrol=0;
npain=0;
nrelief=0;

orders=[
{'Bannantine',2},
{'Barker',1},
{'Barker',2},
{'Carter',0},
{'Chambers',1},
{'Chambers',2},
{'Fosse',0},
{'Gervasi',1},
{'Gervasi',2},
{'Gerwick',0},
{'Gies',1},
{'Gies',2},
{'Kramer',1},
{'Kramer',2},
{'Mitchell',1},
{'Neubauer',1},
{'Neubauer',2},
{'Oak',0},
{'Olivas',1},
{'Palosaari',1},
{'Polosaari',2},	
{'Roth',2},
{'Schneider',0},
{'Simone',1},
{'Sorbo',0}
];

for subject=1:length(orders)
    if orders{subject,2}==0
        ncontrol=ncontrol+1;
    elseif orders{subject,2}==1
        npain=npain+1;
    elseif orders{subject,2}==2
        nrelief=nrelief+1;
    end
end

%%
%Insert a column of order number into start of raw EEG data
for sub=1:n
    orderedData{sub}=[repelem(orders{sub,2},length(data{sub}))' data{sub}]
end

%%
%Sanity check to make sure lengths match in original data and ordered data
for sub=1:n
    if length(data{sub})~=length(orderedData{sub})
        error('Sizes do not match from OG to preprocessed')
    end
end

%%
%exporting data matrix as CSV file, numbered in alphabetical order
for sub=1:n
    csvwrite(strcat(num2str(sub),'.csv'),orderedData{sub})
end