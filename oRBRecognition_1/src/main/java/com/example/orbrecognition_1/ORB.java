package com.example.orbrecognition_1;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.highgui.Highgui;

import android.util.Log;
import android.widget.ImageView;


public class ORB {
	
	private static final String TAG="ORBrecognition:ORB.java";
	
	private byte data[];
	private Mat rgb;
	private Mat image,img;
	
	private Mat descriptors;
	private FeatureDetector detector;
	private MatOfKeyPoint keypoints;
	private DescriptorExtractor descriptor;
	private BufferedInputStream bis;
	private BufferedOutputStream bos;
	
	private File file;
	public void createDB(String filepath, String filename) {
		Log.i(TAG,"createDB");
		detector = FeatureDetector.create(FeatureDetector.ORB);
		descriptor = DescriptorExtractor.create(DescriptorExtractor.ORB);
		keypoints = new MatOfKeyPoint();
		descriptors = new Mat();
		image = Highgui.imread(filepath + "/" + filename + ".jpg", 0);
		
		Log.i(TAG,filepath + "/" + filename + ".jpg");
		
		detector.detect(image, keypoints);
		descriptor.compute(image, keypoints, descriptors);
		data = new byte[(int) (descriptors.total() * descriptors.channels())];
		descriptors.get(0, 0, data);
		
		
		try {
			bos = new BufferedOutputStream(new FileOutputStream(filepath + "/" + filename + ".des"));
			bos.write(data);
			bos.flush();
			bos.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		image.release();
		descriptors.release();
		keypoints.release();
		
		Log.i(TAG,"finish createDB");
		}
	
	

	
	public Mat resdDB(String filepath, String filename) {
		descriptors = new Mat();
		try {
			file = new File(filepath + "/" + filename + ".des");
			data = new byte[(int) file.length()];
			bis = new BufferedInputStream(new FileInputStream(file));
			bis.read(data);
			bis.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		descriptors = new Mat(data.length / 32, 32, CvType.CV_8UC1);
		descriptors.put(0, 0, data);
		Log.i("ImageMat", String.valueOf(descriptors));
		return descriptors;
		
	}
	
	
	public Mat detection(String filepath, String filename){
		detector = FeatureDetector.create(FeatureDetector.ORB);
		descriptor = DescriptorExtractor.create(DescriptorExtractor.ORB);
		keypoints = new MatOfKeyPoint();
		descriptors = new Mat();
		image = Highgui.imread(filepath + "/" + filename, 0);		
		detector.detect(image, keypoints);
		descriptor.compute(image, keypoints, descriptors);
		Log.i("ImageMat", String.valueOf(descriptors));
		return descriptors;
	
	}

}
