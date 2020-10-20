/*
 * Copyright (c) 2019 Skymind AI Bhd.
 * Copyright (c) 2020 CertifAI Sdn. Bhd.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.certifai.training.datavec.loadimage;

import ai.certifai.Helper;
import com.sun.scenario.effect.Crop;
import org.apache.commons.io.FileUtils;
import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.*;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.jfree.data.general.Dataset;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.util.ArchiveUtils;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class LoadImageDemo {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(LoadImageDemo.class);
    private static String dataDir;
    private static String downloadLink;

    private static Random randNumGen = new Random(123);
    private static String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    private static ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

    private static int height = 50;
    private static int width = 50;
    private static int channels = 3;

    private static int batchSize = 24;
    private static int numLabels = 12;

    private static DataNormalization scaler = new ImagePreProcessingScaler(0, 1);

    public static void main(String[] args) throws Exception {
        downloadLink= Helper.getPropValues("dataset.plant.seed.url");;
        dataDir= Paths.get(System.getProperty("user.home"),Helper.getPropValues("dl4j_home.data")).toString();

        //File myFile = new ClassPathResource();

        File parentDir = new File(Paths.get(dataDir,"plant-seedlings-classification","train").toString());
        if(!parentDir.exists()) downloadAndUnzip();

        /*
        Exercise 2: Create image iterator
        - create FileSplit point to images parent folder
        - create random path filter using RandomPathFilter
        - split images into training and test dataset using FileSplit.sample
        - read image using ImageRecordReader
        - define and initialize image transformation
        - create dataset iterator
        - set image data normalization
         */

        //create FileSplit point to images parent folder
        FileSplit fileSplit = new FileSplit(parentDir);

        //create random path filter using RandomPathFilter
        RandomPathFilter RPF = new RandomPathFilter(randNumGen, allowedExtensions);

        //split images into training and test dataset using FileSplit.sample
        InputSplit[] filesinDirSplit = fileSplit.sample(RPF, 70, 30);
        InputSplit trainData = filesinDirSplit[0];
        InputSplit testData = filesinDirSplit[1];

        //read image using ImageRecordReader
        ImageRecordReader trainRR = new ImageRecordReader(height, width, channels, labelMaker);
        ImageRecordReader testRR = new ImageRecordReader(height, width, channels, labelMaker);

        FlipImageTransform horizontalFlip = new FlipImageTransform(1);
        ImageTransform cropImage = new CropImageTransform(5);
        ImageTransform rotateImage = new RotateImageTransform(randNumGen, 15);
        ImageTransform showImage = new ShowImageTransform("Image", 1000);
        boolean shuffle = false;
        List<Pair<ImageTransform, Double>> pipeline = Arrays.asList(
                new Pair<>(horizontalFlip, 0.5),
                new Pair<>(cropImage, 0.5),
                new Pair<>(rotateImage, 0.5),
                new Pair<>(showImage, 1.0)
        );
        ImageTransform transform = new PipelineImageTransform(pipeline,shuffle);

        //create dataset iterator
        trainRR.initialize(trainData, transform);
        testRR.initialize(testData);

        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, batchSize, 1, numLabels);
        DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, batchSize, 1, numLabels);

        //set image data normalization
        trainIter.setPreProcessor(scaler);
        testIter.setPreProcessor(scaler);

        System.out.println(trainIter.next());

        int batchIndex = 0;
        while(trainIter.hasNext())
        {
            DataSet ds = trainIter.next();

            batchIndex += 1;
            System.out.println("\nBatch number: " + batchIndex);
            System.out.println("Feature vector shape: " + Arrays.toString(ds.getFeatures().shape()));
            System.out.println("Label vector shape: " +Arrays.toString(ds.getLabels().shape()));
        }

    }

    private static void downloadAndUnzip() throws IOException {
        String dataPath = new File(dataDir).getAbsolutePath();
        File zipFile = new File(dataPath, "plant-seedings-classification.zip");

        log.info("Downloading the dataset from "+downloadLink+ "...");
        FileUtils.copyURLToFile(new URL(downloadLink), zipFile);

        if(!Helper.getCheckSum(zipFile.getAbsolutePath())
                .equalsIgnoreCase(Helper.getPropValues("dataset.plant.seed.hash"))){
            log.info("Downloaded file is incomplete");
            System.exit(0);
        }

        log.info("Unzipping "+zipFile.getAbsolutePath());
        ArchiveUtils.unzipFileTo(zipFile.getAbsolutePath(), dataPath);
    }
}
