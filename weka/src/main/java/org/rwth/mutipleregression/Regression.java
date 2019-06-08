package org.rwth.mutipleregression;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
/**
 * https://github.com/shuchengc/weka-example
 *
 */
public class Regression {
	
	public static void main(String[] args) throws Exception{
		
		Instances dataset = createDataSet(new File(args[0]));
		/**
		 * linear regression model
		 */
		
		// warm up
		for(int i = 0; i<100; i++) {
			calculate(dataset);
		}
		final int times = 1000;
		LinearRegression lr = null;
		long start = System.currentTimeMillis();
		for(int i = 0; i<times; i++) {
			lr = calculate(dataset);
		}
		long end = System.currentTimeMillis();
		
		System.out.println(String.format("Calculation time: %f ms.", (end - start)/(float)times));
		
		System.out.println(lr);
		Evaluation lreval = new Evaluation(dataset);
		lreval.evaluateModel(lr, dataset);
		System.out.println(lreval.toSummaryString());
		/**
		 * svm regression model
		 */
//		SMOreg smoreg = new SMOreg();
//		smoreg.buildClassifier(dataset);
//		Evaluation svmregeval = new Evaluation(dataset);
//		svmregeval.evaluateModel(smoreg, dataset);
//		System.out.println(svmregeval.toSummaryString());
		
	}

	private static LinearRegression calculate(Instances dataset) throws Exception {
		LinearRegression lr = new LinearRegression();
		lr.buildClassifier(dataset);
		return lr;
	}

	private static Instances createDataSet(File afrrFile) throws FileNotFoundException, Exception {
		DataSource source = new DataSource(new FileInputStream(afrrFile));
		Instances dataset = source.getDataSet();
		dataset.setClassIndex(dataset.numAttributes()-1);
		return dataset;
	}
}
