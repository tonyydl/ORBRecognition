package com.example.orbrecognition_1;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;

import android.app.Activity;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;

public class MainActivity extends Activity {

	private Mat query_descriptors;
	private Mat[] db_descriptors;
	private Mat[] Test_descriptors;
	private FeatureDetector detector;
	private DescriptorExtractor descriptor;
	private DescriptorMatcher matcher;
	private MatOfKeyPoint keypoints;
	private MatOfDMatch matches;
	private int alljpgcount = 0;
	private int alltestjpgcount = 0;
	private Mat mRgba, mGray;
	private Mat ImageMat;
	private Mat Imagetest;
	private int match_points, match_trainnumber, match_testnumber;
	private int thenumber;
	private String[] db_folders;
	private String[] db_names;
	private String[] db_filepath;
	private String[] Test_names;
	private String[] Test_filepath;
	private static final String mobile_path = Environment
			.getExternalStorageDirectory().toString() + "/DB_NEW/ALL";
	private static final String mobile_testpath = Environment
			.getExternalStorageDirectory().toString()
			+ "/Building20forARBuilding";

	// 取得資料夾內jpg的數量
	private int folderjpgcount(File file) {
		File[] list = file.listFiles();
		int count = 0;
		for (File f : list) {
			String name = f.getName();
			if (name.endsWith(".jpg"))
				count++;
		}
		return count;
	}

	// 遞迴取得資料夾內的jpg的數量
	private int folderjpgcount2(File file) {
		int count = 0;
		String name;
		File[] list = file.listFiles();
		for (File f : list)
			// 1 ... 2 ... 3
			if (f.isDirectory()) { // 如果是資料夾
				File[] list2 = f.listFiles();
				for (File ff : list2) {
					name = ff.getName();
					if (name.endsWith(".jpg"))
						count++;
				}
			}
		return count;
	}

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);	
	}

	private static final String TAG = "ORBRecognition_1";

	private BaseLoaderCallback mOpenCVCallBack = new BaseLoaderCallback(this) {
		private int alltrainjpgcount;
		private int trainArr;

		@Override
		public void onManagerConnected(int status) {
			switch (status) {
			case LoaderCallbackInterface.SUCCESS: {
				Log.i(TAG, "OpenCV loaded successfully");
				//
				detector = FeatureDetector.create(FeatureDetector.ORB);
				descriptor = DescriptorExtractor
						.create(DescriptorExtractor.ORB);
				matcher = DescriptorMatcher
						.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
				
				// 各自取得Train與Test的圖片的數量
				int countTrain = folderjpgcount2(new File(mobile_path));
				int countTest = folderjpgcount(new File(mobile_testpath));
				
				// 建立db(train) descriptors陣列及給予長度
				db_descriptors = new Mat[countTrain];
				// 建立train 的資料夾陣列
				db_folders = new String[countTrain];
				// 建立train 的名稱陣列
				db_names = new String[countTrain];
				// 建立train 的file path陣列
				db_filepath = new String[countTrain];
				
				// 建立test descriptors陣列及給予長度
				Test_descriptors = new Mat[countTest];
				// 建立test 的名稱陣列
				Test_names = new String[countTest];
				// 建立test 的file path陣列
				Test_filepath = new String[countTest];
				
				// Train的File Folder
				// 因為Train分成好幾個資料夾，圖資沒有全部放在一起
				File TrainFolder = new File(mobile_path);
				
				//全部Train訓練圖片陣列代號
				trainArr = 0;
				
				//得到Train的Mat
				for (File f: TrainFolder.listFiles()){ //1 ... 2 ... 3
					// 1 , 2 ,3 ,4 資料夾
        			String folder = f.getName();
        		    if(f.isDirectory()){ //如果是資料夾
        		    	for(File ff: f.listFiles()){
        		    		//1.jpg .. 2.jpg .. 3.jpg .. n.jpg .. 
        		    		String name = ff.getName();
        		    		if (name.endsWith(".jpg")){
        		    			//folder path
        		    			String folderpath = mobile_path+"/"+folder;
        		    			//file path
        		    			String filepath = folderpath+"/"+name;
        		    			//如果是jpg就取它的特徵值
        		    			db_descriptors[trainArr] = 
        		    					new ORB().detection(folderpath, name);
        		    			db_folders[trainArr] = folder;
        		    			db_names[trainArr] = name;
        		    			db_filepath[trainArr] = filepath;
        		    			Log.i(TAG,"folderpath:"+folderpath+" name:"+ name+" filepath"+filepath);
        		    			trainArr++;
        		    		}
        		    	}
        		    }
				}
				
				//得到Test的Mat
				for(int i=0;i<countTest;i++){
					Test_names[i] = Integer.toString(i+1)+".jpg";
					Test_filepath[i] = mobile_testpath+"/"+Test_names[i];
					Test_descriptors[i] = new ORB().detection(mobile_testpath, 
							Test_names[i]);
				}			
				
				//總共花費時間
				double totaltime = 0;
				//平均花費時間
				double avgtime = 0;
				
				//開始進行比對
				try {
					FileWriter out = new FileWriter(mobile_testpath+"/ORB_result.txt",
							false);
					BufferedWriter bw = new BufferedWriter(out);
					for (int j = 0; j < Test_descriptors.length; j++) {
						
						//開始計時
						Stopwatch st = new Stopwatch();
						
						match_points = 0;
						match_trainnumber = 0;
						match_testnumber = 0;
	
						for (int i = 0; i < db_descriptors.length; i++) {
							matches = new MatOfDMatch();
							matcher.match(Test_descriptors[j],db_descriptors[i],matches);
							
							int DIST_LIMIT = 45;// 45
							List<DMatch> matchesList = matches.toList();
							List<DMatch> matches_final = new ArrayList<DMatch>();
	
							for (int ii = 0; ii < matchesList.size(); ii++) {
								if (matchesList.get(ii).distance <= DIST_LIMIT) {
									matches_final.add(matches.toList().get(ii));
								}
							}
							if (match_points < matches_final.size()) {
								match_points = matches_final.size();
								match_trainnumber = i;
							}
						}
						
						double endTime = st.elapsedTime();
						totaltime+=endTime;
						String Msg = Test_filepath[j];
						if (match_points > 15) {
//							Msg+=" ==> "+db_filepath[match_trainnumber]
//									+ "--------Success! ";
							bw.write(Msg+"花費: " +endTime+"秒");
							bw.newLine();
						} else {
//							Msg+="--------Fail! ";
							bw.write(Msg+"花費: " +endTime+"秒");
							bw.newLine();
						}
					}
					avgtime = totaltime / Double.valueOf(countTest);
					bw.write("總共花費: "+totaltime+"秒");
					bw.newLine();
					bw.write("平均花費: "+avgtime+"秒");
					bw.close();
				}
				catch (IOException ioe) {
					System.out.print(ioe);
				}
				
				
				
			}
				{
					super.onManagerConnected(status);
				}
				break;
			}
		}
	};

	@Override
	protected void onResume() {
		super.onResume();
		if (!OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_6, this,
				mOpenCVCallBack)) {
			Log.e(TAG, "Cannot connect to OpenCV Manager");
		}
	}

	@Override
	protected void onStop() {
		// TODO Auto-generated method stub
		super.onStop();
	}

	@Override
	protected void onDestroy() {
		// TODO Auto-generated method stub
		super.onDestroy();
	}

}
