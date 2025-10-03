FolderPath = "D:\Workspace\DL for Computer Vision\Data\MathWorks Created\ASL Alphabet\Classification\Train";
ASLimds = imageDatastore(FolderPath, "IncludeSubfolders",true ,"LabelSource","foldernames");

[aslTrain, aslVal] = splitEachLabel(ASLimds, 0.8, "randomized");

resnet50 = imagePretrainedNetwork("resnet50", "NumClasses", 24);
inputSize = resnet50.Layers(1).InputSize;

augTrain = augmentedImageDatastore(inputSize, aslTrain);
augVal = augmentedImageDatastore(inputSize, aslVal);
%%
numberOfTrainingImages = numel(ASLimds.Files)
150*4
iterationsPerEpoch = floor(numberOfTrainingImages / 150)
%%
opts = trainingOptions("adam", "ValidationData",augVal, "Shuffle","every-epoch","Plots","training-progress", "OutputNetwork","best-validation","MiniBatchSize", 150, "ValidationFrequency",iterationsPerEpoch);

ASLClassifier = trainnet(augTrain, resnet50, "crossentropy", opts)
%%
testFolderPath = "D:\Workspace\DL for Computer Vision\Data\MathWorks Created\ASL Alphabet\Classification\Test";
testImds = imageDatastore(testFolderPath, "IncludeSubfolders",true, "LabelSource","foldernames");
length(testImds.Files)
augTest = augmentedImageDatastore(inputSize, testImds);
classNames = categories(testImds.Labels);

testPreds = minibatchpredict(ASLClassifier, augTest);
testPredsClasses = scores2label(testPreds, classNames);

testAccuracy = nnz(testPredsClasses == testImds.Labels) / length(testPredsClasses)
confusionchart(testImds.Labels, testPredsClasses)
%%
img = read(ASLimds);
size(img)
testASLmodel(ASLClassifier)