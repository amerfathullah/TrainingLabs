package ai.certifai.training.convolution.mnist;

import io.vertx.core.logging.Logger;
import io.vertx.core.logging.LoggerFactory;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.fetchers.DataSetType;
import org.deeplearning4j.datasets.iterator.impl.Cifar10DataSetIterator;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;

public class TrainCifar {

    private static final Logger log = LoggerFactory.getLogger(MnistClassifier.class);
    private static final int batchSize = 64;
    private static final int seed = 1234;
    private static final double learningRate = 0.001;
    private static final int outputNum = 10;
    private static final int height = 32;
    private static final int width = 32;
    private static MultiLayerNetwork model = null;
    private static final int nEpochs = 1;

    public static void main(String[] args) throws Exception
    {
        log.info("Data load and vectorization...");
        DataSetIterator cifarTrain = new Cifar10DataSetIterator(batchSize, DataSetType.TRAIN);
        DataSetIterator cifarTest = new Cifar10DataSetIterator(batchSize, DataSetType.TEST);

        System.out.println(cifarTrain.next());

        log.info("Network configuration and training...");

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(learningRate))
                .weightInit(WeightInit.XAVIER)
                .l2(0.0001)
                .list()
                .layer(0, new ConvolutionLayer.Builder()
                        .nIn(3)
                        .nOut(32)
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new ConvolutionLayer.Builder()
                        .nIn(32)
                        .nOut(64)
                        .stride(2, 2)
                        .kernelSize(5, 5)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new SubsamplingLayer.Builder()
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .build())
                .layer(3, new ConvolutionLayer.Builder()
                        .nIn(64)
                        .nOut(outputNum)
                        .stride(1, 1)
                        .kernelSize(3, 3)
                        .activation(Activation.RELU)
                        .build())
                .layer(4, new DenseLayer.Builder()
                        .nOut(100)
                        .activation(Activation.TANH)
                        .build())
                .layer(5, new OutputLayer.Builder()
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(100)
                        .nOut(outputNum)
                        .build())
                .setInputType(InputType.convolutional(height, width, 3)) // InputType.convolutional for normal image
                .backpropType(BackpropType.Standard)
                .build();

        StatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);

        model = new MultiLayerNetwork(conf);
        model.init();

        System.out.println(model.summary());

        model.setListeners(new ScoreIterationListener(10));

        // evaluation while training (the score should go down)
        for (int i = 0; i < nEpochs; i++) {
            model.fit(cifarTrain);

            log.info("Completed epoch {}", i);
            Evaluation eval = model.evaluate(cifarTest);
            log.info(eval.stats());
            cifarTrain.reset();
            cifarTest.reset();
        }

        //LocationToSave model
        File LocationToSave = new File(System.getProperty("java.io.tmpdir"), "/trained_model.zip");

        System.out.println(LocationToSave.toString());
        //Save your model
        ModelSerializer.writeModel(model, LocationToSave, false);

    /*
   #### LAB STEP 2 #####
   Test on a single image
    */
    }

}
