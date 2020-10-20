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

import static org.bytedeco.opencv.global.opencv_core.add;
import static org.bytedeco.opencv.global.opencv_core.subtract;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgproc.GaussianBlur;
/*
* TASKS:
* -------
* 1. Load any image from the Resources folder
* 2. Apply Unsharp Masking by following the steps shown in the Day 6 lecture slides on Unsharp Masking
* 3. Display the follwing:
*       - the input image
*       - the "detail" (residual after removing smoothed image from the input)
*       - the sharpened image
*
* */

import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;

public class UnsharpMasking {
    public static void main(String[] args) throws IOException {

        Mat src = imread(new ClassPathResource("image_processing/lena.png").getFile().getAbsolutePath());

        Display.display(src, "Original");

        Mat smoothed = new Mat();
        Mat detail = new Mat();
        Mat sharpened = new Mat();

        GaussianBlur(src, smoothed, new Size(3,3), 2);

        subtract(src, smoothed, detail);

        add(src, detail, sharpened);

        Display.display(detail, "Detail");
        Display.display(sharpened, "Sharpened");
    }
}
