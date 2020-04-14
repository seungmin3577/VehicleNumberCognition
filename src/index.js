const { Canvas, createCanvas, Image, ImageData, loadImage } = require('canvas');
const { JSDOM } = require('jsdom');
const cv2 = require('opencv.js');
const ffmpeg = require('ffmpeg');
const { writeFileSync } = require('fs');
const Tesseract = require('tesseract.js');
const fs = require('fs');
const vision = require('@google-cloud/vision');
const clientOptions = { apiEndpoint: 'ko-vision.googleapis.com' };

// const path = __dirname +'/public/DJI_0019.mp4'
// const carPath = __dirname + '/public/convert_images'
// const carPath = __dirname + '/public/car_images'
const client = new vision.ImageAnnotatorClient();

const MIN_AREA = 70;
const MIN_WIDTH = 2;
const MIN_HEIGHT = 8;
const MIN_RATIO = 0.25;
const MAX_RATIO = 1.0;

const MAX_DIAG_MULTIPLYER = 2;
const MAX_ANGLE_DIFF = 12.0;
const MAX_AREA_DIFF = 0.5;
const MAX_WIDTH_DIFF = 0.8;
const MAX_HEIGHT_DIFF = 0.2;
const MIN_N_MATCHED = 3;

const PLATE_WIDTH_PADDING = 1.3;
const PLATE_HEIGHT_PADDING = 1.3;

const MIN_PLATE_RATIO = 3;
const MAX_PLATE_RATIO = 10;

const SperateMovie = async path => {
  try {
    await new ffmpeg(path, function(err, video) {
      if (!err) {
        video.fnExtractFrameToJPG(
          __dirname + '/public/convert_images',
          {
            every_n_frames: 10,
            file_name: '%s',
            keep_pixel_aspect_ratio: true,
            keep_aspect_ratio: true,
          },
          async function(error, files) {
            console.log('The video is ready to be processed');
            if (!error) {
              for (const file of files) {
                await imageRecognize(file);
              }
            } else {
              console.log('error', error);
            }
          },
        );
      } else {
        console.log('Error: ' + err);
      }
    });
  } catch (e) {
    console.log('SperateMovie -> e', e);
  }
};

const imageRecognize = async fileName => {
  //Inital Dom
  let dom = new JSDOM();
  global.document = dom.window.document;
  // The rest enables DOM image and canvas and is provided by node-canvas
  global.Image = Image;
  global.HTMLCanvasElement = Canvas;
  global.ImageData = ImageData;
  global.HTMLImageElement = Image;

  console.log('start image Recoginize of ' + fileName);
  try {
    let canvas = createCanvas();

    const src = await loadImage(fileName);
    const originalImage = cv2.imread(src);
    let grayImage = new cv2.Mat();
    cv2.imshow(canvas, originalImage);
    cv2.cvtColor(originalImage, grayImage, cv2.COLOR_BGR2GRAY, 0);
    cv2.imshow(canvas, grayImage);
    console.log('Convert to gray color image complete!');

    let bulrImage = new cv2.Mat();
    let ksize = new cv2.Size(5, 5);
    cv2.GaussianBlur(grayImage, bulrImage, ksize, 0);
    // grayImage.delete();
    cv2.imshow(canvas, bulrImage);
    console.log('Convert to blur image complete!');
    const thresholdImage = new cv2.Mat();
    cv2.adaptiveThreshold(
      bulrImage,
      thresholdImage,
      255,
      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
      cv2.THRESH_BINARY_INV,
      19,
      9,
    );
    // bulrImage.delete();
    cv2.imshow(canvas, thresholdImage);
    console.log('Convert to threshold image complete!');

    let contoursImage = cv2.Mat.zeros(
      originalImage.rows,
      originalImage.cols,
      cv2.CV_8UC3,
    );
    let contours = new cv2.MatVector();
    let hierarchy = new cv2.Mat();
    // You can try more different parameters
    cv2.findContours(
      thresholdImage,
      contours,
      hierarchy,
      cv2.RETR_CCOMP,
      cv2.CHAIN_APPROX_SIMPLE,
    );
    // thresholdImage.delete();

    // draw contours with random Scalar
    for (let i = 0; i < contours.size(); ++i) {
      let color = new cv2.Scalar(
        Math.round(255),
        Math.round(255),
        Math.round(255),
      );
      cv2.drawContours(
        contoursImage,
        contours,
        i,
        color,
        1,
        cv2.LINE_8,
        hierarchy,
        100,
      );
    }
    cv2.imshow(canvas, contoursImage);
    // writeFileSync('contours.jpg', canvas.toBuffer('image/jpeg'));
    console.log('Convert to contours image complete!');

    const reactImage = cv2.Mat.zeros(
      originalImage.rows,
      originalImage.cols,
      cv2.CV_8UC3,
    );
    let contoursDict = [];

    for (let i = 0; i < contours.size(); i++) {
      const { x, y, width, height } = cv2.boundingRect(contours.get(i));
      let point1 = new cv2.Point(x, y);
      let point2 = new cv2.Point(x + width - 1, y + height);
      let rectangleColor = new cv2.Scalar(255, 255, 255);

      cv2.rectangle(reactImage, point1, point2, rectangleColor, 2);
      contoursDict.push({
        contour: contours.get(i),
        x,
        y,
        width,
        height,
        cx: x + width / 2,
        cy: y + height / 2,
      });
    }

    cv2.imshow(canvas, reactImage);
    writeFileSync('rect.jpg', canvas.toBuffer('image/jpeg'));
    console.log('Convert to reactangle image complete!');

    let possibleContours = [];
    let count = 0;

    for (let i = 0; i < contoursDict.length; i++) {
      const dict = contoursDict[i];

      const area = dict['width'] * dict['height'];
      const ratio = dict['width'] / dict['height'];

      const condition1 = area > MIN_AREA;
      const condition2 = dict['width'] > MIN_WIDTH;
      const condition3 = dict['height'] > MIN_HEIGHT;
      const condition4 = MIN_RATIO < ratio;
      const condition5 = MAX_RATIO > ratio;

      if (condition1 && condition2 && condition3 && condition4 && condition5) {
        dict['idx'] = count;
        count += 1;
        possibleContours.push(dict);
      }
    }

    const sortRectImage = cv2.Mat.zeros(
      originalImage.rows,
      originalImage.cols,
      cv2.CV_8UC3,
    );

    for (let i = 0; i < possibleContours.length; i++) {
      const dict = possibleContours[i];
      let point1 = new cv2.Point(dict['x'], dict['y']);
      let point2 = new cv2.Point(
        dict['x'] + dict['width'],
        dict['y'] + dict['height'],
      );
      let rectangleColor = new cv2.Scalar(255, 255, 255);

      cv2.rectangle(sortRectImage, point1, point2, rectangleColor, 2);
    }
    cv2.imshow(canvas, sortRectImage);
    writeFileSync('sortRect.jpg', canvas.toBuffer('image/jpeg'));

    const resultIndex = findCharactors(possibleContours);

    let matchedResult = [];
    resultIndex.forEach(indexList => {
      matchedResult.push(
        indexList.map(index => {
          return possibleContours[index];
        }),
      );
    });

    let resultRectImage = cv2.Mat.zeros(
      originalImage.rows,
      originalImage.cols,
      cv2.CV_8UC3,
    );

    matchedResult.forEach(result => {
      result.forEach(dict => {
        let point1 = new cv2.Point(dict['x'], dict['y']);
        let point2 = new cv2.Point(
          dict['x'] + dict['width'],
          dict['y'] + dict['height'],
        );
        let rectangleColor = new cv2.Scalar(255, 255, 255);
        cv2.rectangle(resultRectImage, point1, point2, rectangleColor, 2);
      });
    });

    cv2.imshow(canvas, resultRectImage);
    writeFileSync('resultRect.jpg', canvas.toBuffer('image/jpeg'));
    console.log('Convert to result rectangle image complete!');

    let plateImgs = [];
    let plateInfos = [];

    for (let i = 0; i < matchedResult.length; i++) {
      const matchedChars = matchedResult[i];
      const sortedChars = matchedChars.sort(
        (first, second) => first['cx'] - second['cx'],
      );

      const firstChar = sortedChars[0];
      const lastChar = sortedChars[matchedChars.length - 1];

      let plateCx = (firstChar['cx'] + lastChar['cx']) / 2;
      let plateCy = (firstChar['cy'] + lastChar['cy']) / 2;

      let plateWidth =
        (lastChar['x'] + lastChar['width'] - firstChar['x']) *
        PLATE_WIDTH_PADDING;

      let sumHeight = 0;
      sortedChars.forEach(dict => {
        sumHeight += dict['height'];
      });
      let plateHeight = (sumHeight / sortedChars.length) * PLATE_HEIGHT_PADDING;

      let triangleHeight = firstChar['cy'] - lastChar['cy'];
      let triangleHypotenus = Math.sqrt(
        Math.pow(lastChar['cx'] - firstChar['cx'], 2) +
          Math.pow(lastChar['cy'] - firstChar['cy'], 2),
      );

      const pi = Math.PI;
      let angle = Math.atan(triangleHeight / triangleHypotenus) * (180 / pi);
      let rotationMatrix = cv2.getRotationMatrix2D(
        new cv2.Point(plateCx, plateCy),
        -angle,
        1,
      );

      canvas = createCanvas();

      let plateImage = new cv2.Mat();
      let plateGrayImage = new cv2.Mat();
      let plateBlurImage = new cv2.Mat();

      let dsize = new cv2.Size(
        Math.abs(plateWidth) * 1,
        Math.abs(plateHeight) * 1,
      );
      let M = cv2.matFromArray(2, 3, cv2.CV_64FC1, [
        1,
        0,
        (plateCx - plateWidth / 2) * -1,
        0,
        1,
        (plateCy - plateHeight / 2) * -1,
      ]);
      // You can try more different parameters
      cv2.warpAffine(
        originalImage,
        plateImage,
        rotationMatrix,
        new cv2.Size(originalImage.cols, originalImage.rows),
      );
      cv2.imshow(canvas, plateImage);
      console.log('Complete Crop Image.');
      cv2.cvtColor(plateImage, plateGrayImage, cv2.COLOR_BGR2GRAY, 0);
      cv2.imshow(canvas, plateGrayImage);
      console.log('CropImage convert to gray');
      cv2.GaussianBlur(plateGrayImage, plateBlurImage, ksize, 0);
      cv2.adaptiveThreshold(
        plateBlurImage,
        plateBlurImage,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        19,
        9,
      );
      const croppedImage = cv2.warpAffine(
        plateBlurImage,
        plateImage,
        M,
        dsize,
        cv2.INTER_LINEAR,
        cv2.BORDER_CONSTANT,
        new cv2.Scalar(),
      );
      cv2.imshow(canvas, plateImage);
      writeFileSync(`carNumber.${i}.jpg`, canvas.toBuffer('image/jpeg'));
      console.log('Convert to plate image complete!');

      plateImgs.push(croppedImage);
      plateInfos.push({
        x: plateCx - plateWidth / 2,
        y: plateCy - plateHeight / 2,
        width: plateWidth,
        height: plateHeight,
      });
    }

    let carNumber = [];
    let files = fs.readdirSync('.');
    let result = await Promise.all(
      files.map(async item => {
        if (item.includes('carNumber')) {
          const [result] = await client.documentTextDetection('./' + item);
          const detections = result.fullTextAnnotation;
          console.log('imageRecognize -> detections', detections);
          if (
            detections === undefined ||
            detections === null ||
            detections === ''
          ) {
            return '';
          } else {
            return detections.text;
          }

          // return Tesseract.recognize(
          // 	'./' + item,
          // 	'kor',
          // 	{
          // 		// logger: m => console.log(m)
          // 	}
          // ).catch(error=> {
          // 	return ;
          // });
        }
      }),
    );

    result = result.filter(item => item !== undefined);
    result.forEach(item => {
      let textConvert;
      if (item !== undefined || item !== null || item !== '') {
        let text;
        console.log('imageRecognize -> typeof(item)', typeof item);
        if (typeof item === 'object') {
          text = item.data.text;
          console.log('imageRecognize -> object item', item.data.text);
        } else {
          text = item;
          console.log('imageRecognize -> text item', item);
        }

        const charactorFilter = /^[ㄱ-ㅎ|가-힣|0-9|\*]+$/;

        if (charactorFilter.test(text)) {
          textConvert = text;
        } else {
          console.log('imageRecognize -> text', text);
          textConvert = text
            .replace('\n', '')
            .replace(/[^0-9|^ㄱ-ㅎ|^가-힣]/g, '')
            .replace(' ', '');
        }

        carNumber.push(textConvert);
      }
    });

    console.log('imageRecognize -> carNumber', carNumber);
    if (carNumber.length > 0) {
      fs.appendFileSync('./text.txt', '\n', 'utf8');
    }
    carNumber.reverse().forEach((item, index) => {
      console.log('imageRecognize -> item', item);
      fs.appendFileSync('./text.txt', item, 'utf8');
      if (index % 2 === 1) {
        fs.appendFileSync('./text.txt', '\n', 'utf8');
      }
    });

    await deleteCarNumberFiles();

    // Matrix reset
    originalImage.delete();
    grayImage.delete();
    bulrImage.delete();
    thresholdImage.delete();
    contoursImage.delete();
    contours.delete();
    hierarchy.delete();
    reactImage.delete();
    sortRectImage.delete();
    resultRectImage.delete();
    // plateImage.delete();
    // plateGrayImage.delete();
    // plateBlurImage.delet();
    // croppedImage.delete();

    //dom reset
    dom = undefined;
    global.document = undefined;
    // The rest enables DOM image and canvas and is provided by node-canvas
    global.Image = undefined;
    global.HTMLCanvasElement = undefined;
    global.ImageData = undefined;
    global.HTMLImageElement = undefined;

    return;
  } catch (error) {
    console.log('imageRecognize -> error', error);
    return;
  }
};

const findCharactors = contourList => {
  const matchedResultIndex = [];
  const contourList1 = contourList.filter(item => {
    if (item !== undefined) {
      return item;
    }
  });
  const contourList2 = contourList.filter(item => {
    if (item !== undefined) {
      return item;
    }
  });

  for (let i = 0; i < contourList1.length; i++) {
    const dict1 = contourList1[i];
    const matchedContoursIndex = [];
    for (let j = 0; j < contourList2.length; j++) {
      const dict2 = contourList2[j];
      if (dict1['idx'] == dict2['idx']) {
        continue;
      }

      const pi = Math.PI;
      const dx = Math.abs(dict1['cx'] - dict2['cx']);
      const dy = Math.abs(dict1['cy'] - dict2['cy']);
      let angle_diff;
      let area_diff;
      let width_diff;
      let height_diff;

      const diagonalLength1 = Math.sqrt(
        Math.pow(dict1['width'], 2) + Math.pow(dict1['height'], 2),
      );
      // const distance = math.norm([[dict1['cx'], dict1['cy']]] - [[[dict2['cx'], dict2['cy']]]]);
      const distance = Math.sqrt(
        Math.pow(dict1['cx'] - dict2['cx'], 2) +
          Math.pow(dict2['cy'] - dict1['cy'], 2),
      );
      if (dx === 0) {
        angle_diff = 90;
      } else {
        angle_diff = Math.atan2(dy / dx) * (pi / 180);
      }

      area_diff =
        Math.abs(
          dict1['width'] * dict1['height'] - dict2['width'] * dict2['height'],
        ) /
        (dict1['width'] * dict1['height']);
      width_diff = Math.abs(dict1['width'] - dict2['width']) / dict1['width'];
      height_diff =
        Math.abs(dict1['height'] - dict2['height']) / dict1['height'];

      let conditionCount = 0;
      const condition1 = distance < diagonalLength1 * MAX_DIAG_MULTIPLYER; // 대각거리 3배이내로 있는가?
      const condition2 = angle_diff < MAX_ANGLE_DIFF;
      const condition3 = area_diff < MAX_AREA_DIFF;
      const condition4 = width_diff < MAX_WIDTH_DIFF;
      const condition5 = height_diff < MAX_HEIGHT_DIFF;
      condition1 ? (conditionCount += 1) : null;
      condition2 ? (conditionCount += 1) : null;
      condition3 ? (conditionCount += 1) : null;
      condition4 ? (conditionCount += 1) : null;
      condition5 ? (conditionCount += 1) : null;

      if (condition1 && conditionCount >= MIN_N_MATCHED + 1) {
        matchedContoursIndex.push(dict2['idx']);
      }
    }
    matchedContoursIndex.push(dict1['idx']);

    if (matchedContoursIndex.length < MIN_N_MATCHED + 1) {
      continue;
    }

    matchedResultIndex.push(matchedContoursIndex);

    const unmatchedContourIndex = [];
    contourList.forEach(dict_4 => {
      if (dict_4 === undefined) return;
      if (!matchedContoursIndex.includes(dict_4['idx'])) {
        unmatchedContourIndex.push(dict_4['idx']);
      }
    });

    const unmatchedContour = unmatchedContourIndex.map(item => {
      return contourList[item];
    });

    const recursiveContourList = findCharactors(unmatchedContour);

    recursiveContourList.forEach(idx => {
      matchedResultIndex.push(idx);
    });

    break;
  }
  return matchedResultIndex;
};

const deleteCarNumberFiles = () => {
  const dirNames = fs.readdirSync('./');

  dirNames.forEach(item => {
    if (item.includes('carNumber')) {
      fs.unlinkSync(item);
    }
  });

  console.log('deleteCarNumberFiles -> All files delete done.');
};

const deleteConvertFiles = file => {
  fs.appendFileSync('./text.txt', `Start delete ${file}.\n`, 'utf8');
  const dirNames = fs.readdirSync('./src/public/convert_images');

  dirNames.forEach(item => {
    console.log('deleteConvertFiles -> item', item);
    fs.unlinkSync('./src/public/convert_images/' + item);
  });

  fs.appendFileSync('./text.txt', `End delete ${file}.\n`, 'utf8');
};

const recognizeAllFiles = async () => {
  const files = fs.readdirSync(__dirname + '/public/');

  for (const file of files) {
    if (file.includes('mp4')) {
      fs.appendFileSync('./text.txt', `Ready Recognize ${file}.\n`, 'utf8');
      await deleteConvertFiles(file);
      fs.appendFileSync('./text.txt', `Start Recognize ${file}.\n`, 'utf8');
      await SperateMovie(__dirname + '/public/' + file);
      fs.appendFileSync('./text.txt', `End Recognize ${file}.\n`, 'utf8');
    }
  }

  // await SperateMovie(__dirname + '/public/DJI_0017.mp4');
  // await SperateMovie(__dirname + '/public/DJI_0018.mp4');
  // await SperateMovie(__dirname + '/public/DJI_0019.mp4');
  // await SperateMovie(__dirname + '/public/DJI_0020.mp4');
};

recognizeAllFiles();
// SperateMovie(__dirname +'/public/DJI_0017.mp4');
// SperateMovie(__dirname +'/public/DJI_0018.mp4');
// SperateMovie(__dirname +'/public/DJI_0019.mp4');
// SperateMovie(__dirname +'/public/DJI_0020.mp4');

// imageRecognize('./src/public/convert_images/1920x1080_135.jpg');
// imageRecognize('./src/public/car_images/1.jpg');
// imageRecognize('./src/public/car_images/2.jpg');
// imageRecognize('./src/public/car_images/3.jpg');
// imageRecognize('./src/public/car_images/4.jpg');
