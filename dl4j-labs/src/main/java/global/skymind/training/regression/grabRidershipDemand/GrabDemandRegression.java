package global.skymind.training.regression.grabRidershipDemand;/*
 *
 *  * ******************************************************************************
 *  *  * Copyright (c) 2019 Skymind AI Bhd.
 *  *  * Copyright (c) 2020 CertifAI Sdn. Bhd.
 *  *  *
 *  *  * This program and the accompanying materials are made available under the
 *  *  * terms of the Apache License, Version 2.0 which is available at
 *  *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *  *
 *  *  * Unless required by applicable law or agreed to in writing, software
 *  *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  *  * License for the specific language governing permissions and limitations
 *  *  * under the License.
 *  *  *
 *  *  * SPDX-License-Identifier: Apache-2.0
 *  *  *****************************************************************************
 *
 *
 */

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

// TO TRY
//        System.out.println( (latlong.getLat() + 90)*180 + latlong.getLon());

public class GrabDemandRegression {
    public static final int seed = 12345;
    public static final double learningRate = 0.01;
    public static final int nEpochs = 10;
    public static final int batchSize = 1000;
    public static final int nTrain = 2500000;
    
    public static void main(String[] args) throws IOException, InterruptedException  {

        /*
        *  STEP 1: DATA PREPARATION
        *
        * */
        File inputFile = new File(System.getProperty("user.home"), ".deeplearning4j/data/grab/Traffic Management/train/train.csv");

        CSVRecordReader csvRR = new CSVRecordReader(1,',');
        csvRR.initialize(new FileSplit(inputFile));

        Schema inputDataSchema = new Schema.Builder()
                .addColumnString("geohash6")
                .addColumnInteger("day")
                .addColumnString("timestamp")
                .addColumnFloat("demand")
                .build();

        Pattern REPLACE_PATTERN = Pattern.compile("\\:\\d+");

        Map<String,String> map = new HashMap<>();
        map.put(REPLACE_PATTERN.toString(), "");

        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
                .replaceStringTransform("timestamp", map)
                .convertToInteger("timestamp")
                .transform(new GeohashtoLatLonTransform.Builder("geohash6")
                        .addLatDerivedColumn("latitude")
                        .addLonDerivedColumn("longitude").build())
                .removeColumns("geohash6")
                .build();


        //After executing all of these operations, we have a new and different schema:
        Schema outputSchema = tp.getFinalSchema();
        System.out.println(outputSchema);

        //Process the data:
//        List<List<Writable>> originalData = new ArrayList<>();
        List<List<Writable>> trainData = new ArrayList<>();
        List<List<Writable>> valData = new ArrayList<>();
        int i = 0;
        while(csvRR.hasNext()){
            if (i < nTrain) {trainData.add(csvRR.next());}
            else {valData.add(csvRR.next());}
            i ++ ;
        }

        List<List<Writable>> processedDataTrain = LocalTransformExecutor.execute(trainData, tp);
        List<List<Writable>> processedDataVal = LocalTransformExecutor.execute(valData, tp);

        //Create iterator from processedData
        RecordReader collectionRecordReaderTrain = new CollectionRecordReader(processedDataTrain);
        RecordReader collectionRecordReaderVal = new CollectionRecordReader(processedDataVal);

        DataSetIterator trainIter = new RecordReaderDataSetIterator(collectionRecordReaderTrain,batchSize,4,4,true);
        DataSetIterator valIter = new RecordReaderDataSetIterator(collectionRecordReaderVal, processedDataVal.size(),4,4,true);


        /*
         *  STEP 2: MODEL TRAINING
         *
         * */
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainIter);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
        while (trainIter.hasNext()) {
            normalizer.transform(trainIter.next());
        }
        while (valIter.hasNext()) {
            normalizer.transform(valIter.next());
        }
//        normalizer.transform(trainIter);     //Apply normalization to the training data
//        normalizer.transform(valIter);         //Apply normalization to the val data

        //Create the network
        int numInput = 4;
        int numOutputs = 1;
        int nHidden = 15;
        MultiLayerNetwork net = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(learningRate, 0.9))
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInput).nOut(nHidden)
                        .activation(Activation.TANH)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(nHidden).nOut(numOutputs).build())
                .build()
        );
        net.init();
        net.setListeners(new ScoreIterationListener(10));

        StatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);

        trainIter.reset();
        net.fit(trainIter, nEpochs);

        valIter.reset();
        RegressionEvaluation eval = net.evaluateRegression(valIter);
        System.out.println(eval.stats());


        /*
         *  STEP 3: SAVE MODEL FOR TESTING
         *
         * */
        // Where to save model
        File locationToSave = new File(System.getProperty("java.io.tmpdir"), "/trained_regression_model.zip");
        System.out.println(locationToSave.toString());

        // boolean save Updater
        boolean saveUpdater = false;

        // ModelSerializer needs modelname, saveUpdater, Location
        ModelSerializer.writeModel(net,locationToSave,saveUpdater);

    }
}
