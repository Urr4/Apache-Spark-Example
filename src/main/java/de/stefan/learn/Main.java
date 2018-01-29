package de.stefan.learn;

import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.MinMaxScaler;
import org.apache.spark.ml.feature.MinMaxScalerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;

import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.Scanner;

/**
 * @project ApacheSpark
 * Created by @author sschubert on 26.01.2018
 */
public class Main {

    private static final Logger LOGGER = LogManager.getLogger(Main.class);

    public static void main(String[] args) {

        LocalDateTime start = LocalDateTime.now();

        // entrance to spark logic
        executeCalculations();

        // execution tracing
        LocalDateTime end = LocalDateTime.now();
        long timeBetween = ChronoUnit.MILLIS.between(start, end);

        LOGGER.info("Execution took: " + timeBetween + "ms");

        // LOCAL only, keeps Spark-UI alive
        Scanner scan = new Scanner(System.in);
        scan.nextLine();

    }

    private static void executeCalculations() {
        String inputFile = "src/main/resources/Global Superstore Orders.csv";

        SparkSession spark = SparkSession
                .builder()
                .master("local[*]")
                .appName("SparkJob")
                .getOrCreate();

        // should be equal to the number of tasks or a multiple of it, depends on the size of the dataset
        spark.conf().set("spark.sql.shuffle.partitions", 8);

        //Read Data
        Dataset<Row> dataset = spark.read().format("csv")
                .option("sep", ";")
                .option("header", "true")
                .option("inferSchema", "true")
                .option("nullValue", null)
                .option("nanValue", "NA")
                .load(inputFile);

        //Reduce Data to desired columns
        Dataset<Row> reducedByColumns = dataset.select("Order Date", "Sales", "Profit");

        //Add column with date transformed to timestamp
        Column order_long = functions.to_timestamp(functions.col("Order Date"), "dd.MM.yy").cast(DataTypes.LongType);
        Dataset<Row> timestamps = reducedByColumns.withColumn("Order Date", order_long);

        Column replace1 = functions.regexp_replace(functions.col("Sales"), "\\$", "");
        Column replace2 = functions.regexp_replace(functions.col("Sales"), "\\.", "");
        Column replace3 = functions.regexp_replace(functions.col("Sales"), "\\,", ".").cast(DataTypes.DoubleType);

        Column replace4 = functions.regexp_replace(functions.col("Profit"), "\\$", "");
        Column replace5 = functions.regexp_replace(functions.col("Profit"), "\\.", "");
        Column replace6 = functions.regexp_replace(functions.col("Profit"), "\\,", ".").cast(DataTypes.DoubleType);

        //Transform Sales values to double
        Dataset<Row> dataset1 = timestamps.withColumn("Sales", replace1).withColumn("Sales", replace2).withColumn("Sales", replace3);

        //Transform Profit values to double
        Dataset<Row> dataset2 = dataset1.withColumn("Profit", replace4).withColumn("Profit", replace5).withColumn("Profit", replace6);

        VectorAssembler vectorAssembler = new VectorAssembler()
                .setInputCols(new String[]{"Order Date", "Sales"})
                .setOutputCol("features");

        //Add features column holdinf Order Date and Sales
        Dataset<Row> dataset3 = vectorAssembler.transform(dataset2);

        //Split into 60% training and 40% testdata
        double[] splitRatio = {0.6, 0.4};
        Dataset<Row>[] splitted = dataset3.randomSplit(splitRatio);
        Dataset<Row> training = splitted[0];
        Dataset<Row> test = splitted[1];

        //Build linear regression model
        LinearRegression linearRegression = new LinearRegression()
                .setMaxIter(100)
                .setRegParam(0.3)
                .setElasticNetParam(0.8)
                .setLabelCol("Profit");

        //Train model
        LinearRegressionModel linearRegressionModel = linearRegression.fit(training);

        System.out.println("Coefficients: "
                + linearRegressionModel.coefficients() + " Intercept: " + linearRegressionModel.intercept());

        //Apply model to testdata
        Dataset<Row> transform = linearRegressionModel.transform(test);

        //Evaluate Testresult
        RegressionEvaluator regressionEvaluator = new RegressionEvaluator()
                .setLabelCol("Profit")
                .setPredictionCol("prediction");
        double evaluate = regressionEvaluator.evaluate(transform);
        LOGGER.info("Evaluate " + evaluate);

    }

}
