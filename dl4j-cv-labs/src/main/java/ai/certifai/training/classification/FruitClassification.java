package ai.certifai.training.classification;

import ai.certifai.solution.segmentation.CustomLabelGenerator;
import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.ImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.*;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class FruitClassification {

    private static final String[] allowedExtension = BaseImageLoader.ALLOWED_FORMATS;
    private static final int nChannel = 3;
    private static final int width = 60;
    private static final int height = 60;

    private static final int seed = 123;
    private static final int batchSize = 6;
    private static final double lr = 1e-4;
    private static final int nEpoch = 10;
    private static Random randNumGen = new Random(seed);

    public static void main(String[] args) throws IOException {

        //Get your dataset
        File parentDir = new ClassPathResource("fruits").getFile();

        //Create a file split
        FileSplit fileSplit = new FileSplit(parentDir, allowedExtension);

        //Create labels
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        //Path filter
        RandomPathFilter RPF = new RandomPathFilter(randNumGen, allowedExtension);

        //Split dataset into train and test
        InputSplit[] filesInDirSplit = fileSplit.sample(RPF, 70, 30);
        InputSplit trainData = filesInDirSplit[0];
        InputSplit testData = filesInDirSplit[1];

        //Define and execute image transformation
        FlipImageTransform horizontalFlip = new FlipImageTransform(6);
        ImageTransform cropImage = new CropImageTransform(5);
        ImageTransform rotateImage = new RotateImageTransform(randNumGen, 20);
        ImageTransform showImage = new ShowImageTransform("Image", 1000);
        List<Pair<ImageTransform, Double>> pipeline = Arrays.asList(
                new Pair<>(horizontalFlip, 0.4),
                new Pair<>(cropImage, 0.4),
                new Pair<>(rotateImage, 0.4),
                new Pair<>(showImage, 1.0)
        );
        ImageTransform transform = new PipelineImageTransform(pipeline, false);

        //Create and initialize your record reader
        ImageRecordReader trainRR = new ImageRecordReader(height, width, nChannel, labelMaker);
        ImageRecordReader testRR = new ImageRecordReader(height,width, nChannel, labelMaker);

        trainRR.initialize(trainData, transform);
        testRR.initialize(testData);

        //Create train and test iterator
        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, batchSize, 1, 2);
        DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, batchSize, 1, 2);

        //Create model configuration
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(lr))
                .list()
                .layer(new ConvolutionLayer.Builder()
                        .nIn(nChannel)
                        .nOut(16)
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder()
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new ConvolutionLayer.Builder()
                        .nOut(32)
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder()
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder()
                        .nOut(2)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .build())
                .setInputType(InputType.convolutional(height, width, nChannel))
                .build();

        //Build and initialize your model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        System.out.println(model.summary());

        //Set training UI and set listeners
        UIServer uiServer = UIServer.getInstance();
        StatsStorage storage = new InMemoryStatsStorage();
        uiServer.attach(storage);

        model.setListeners(new ScoreIterationListener(10));
        model.setListeners(new StatsListener(storage, 10));

        //Model training
        model.fit(trainIter, nEpoch);

        //Evaluate your model
        Evaluation evalTrain = model.evaluate(trainIter);
        Evaluation evalTest = model.evaluate(testIter);

        System.out.println("Training Set Evaluation" + evalTrain);
        System.out.println("Test Set Evaluation" + evalTest);
    }
}
