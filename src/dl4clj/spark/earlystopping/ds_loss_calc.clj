(ns dl4clj.spark.earlystopping.ds-loss-calc
  (:import [org.deeplearning4j.spark.earlystopping SparkDataSetLossCalculator]))

(defn new-spark-ds-loss-calculator
  "Score calculator to calculate the total loss for the MLN
  on the provided JavaRDD data set (test set)

  :test-rdd (JavaRDD), dataset to calc the score for

  :average? (boolean), Whether to return the average (sum of loss / N),
                       or just the sum of the loss

  :spark-context (org.apache.spark.SparkContext), the spark context"
  [& {:keys [test-rdd average? spark-context]}]
  (SparkDataSetLossCalculator. test-rdd average? spark-context))

(defn calculate-score-spark
  "Calculate the score for the given MLN"
  [& {:keys [ds-loss-calc mln]}]
  (.calculateScore ds-loss-calc mln))
