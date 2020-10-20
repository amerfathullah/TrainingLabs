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

package ai.certifai.training.image_processing;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.IOException;

import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

/*
 *
 * 1. Go to https://image.online-convert.com/, convert resources/image_processing/opencv.png into the following format:
 *       - .bmp
 *       - .jpg
 *       - .tiff
 *     Save them to the same resources/image_processing folder.
 *
 *  2. Use the .imread function to load each all the images in resources/image_processing,
 *       and display them using Display.display
 *
 *
 *  3. Print the following image attributes:
 *       - depth
 *       - number of channel
 *       - width
 *       - height
 *
 *  4. Repeat step 2 & 3, but this time load the images in grayscale
 *
 *  5. Resize file
 *
 *  6. Write resized file to disk
 *
 * */

public class LoadImages {
    public static void main(String[] args) throws IOException {

        //Load image - COLOR
        String imgpath = new ClassPathResource("image_processing/opencv.png").getFile().getAbsolutePath();
        Mat src = imread(imgpath,IMREAD_GRAYSCALE);

        Display.display(src, "Input");

        System.out.println("Number of channels: " + src.depth());
        System.out.println("Number of channels: " + src.channels());
        System.out.println("Number of channels: " + src.arrayWidth());
        System.out.println("Number of channels: " + src.arrayHeight());

        //Image resizing
        Mat dest = new Mat();
        Mat dest_up_linear = new Mat();
        Mat dest_up_nearest = new Mat();
        Mat dest_up_cubic = new Mat();

        // Downsampling
        // Upsampling using diff. interpolation methods
        resize(src, dest, new Size(300, 300)); //DOWNSIZE
        //resize(src, dest_up_linear, new Size(), 2, 2, INTER_LINEAR); //UPSIZE by 2 times of width and height
        resize(src, dest_up_linear, new Size(1478, 1200), 0, 0, INTER_LINEAR); //UPSIZE
        resize(src, dest_up_nearest, new Size(1478, 1200), 0, 0, INTER_NEAREST);
        resize(src, dest_up_cubic, new Size(1478, 1200), 0, 0, INTER_CUBIC);

        //Display resized images
        Display.display(dest, "Downsized");
        Display.display(dest_up_linear, "Upsized - Linear Interpolation");
        Display.display(dest_up_nearest, "Upsized - Nearest Neighbors");
        Display.display(dest_up_cubic, "Upsized - Cubic Interpolation");

        // Write image to disk
        String imgsavepath = imgpath.replace("opencv.tiff", "opencv_small.jpg");
        System.out.println(imgsavepath);
        imwrite(imgsavepath, dest);
    }
}
