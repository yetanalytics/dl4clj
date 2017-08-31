(ns dl4clj.spark.api.dl4j-multi-layer
  (:import [org.deeplearning4j.spark.impl.multilayer SparkDl4jMultiLayer])
  (:require [dl4clj.utils :refer [contains-many?]]))

(defn spark-calc-score
  "Calculate the score for all examples in the provided rdd

  :rdd (JavaRDD<org.nd4j.linalg.dataset.DataSet>) the rdd containing the data
   - can also be RDD<org.nd4j.linalg.dataset.DataSet>

  :average? (boolean), determines if score is calculated via averaging or summing
   over the entire dataset

  :mini-batch-size (int), the size of the mini batches within the rdd
   - should only be supplied when :rdd is a JavaRDD and then still optional"
  [& {:keys [spark-mln rdd average? mini-batch-size]}]
  (if mini-batch-size
    (.calculateScore spark-mln rdd average? mini-batch-size)
    (.calculateScore spark-mln rdd average?)))

(defn eval-classification-spark-mln
  "Evaluate the network in a distributed manner on the provided data

  :rdd (JavaRDD<org.nd4j.linalg.dataset.DataSet>) the rdd containing the data
   - can also be RDD<org.nd4j.linalg.dataset.DataSet>

  :labels (coll), a collection of strings which serve as the labels in evaluation

  :batch-size (int), the batch size for the data within the rdd

   - required args: :spark-mln, :rdd"
  [& {:keys [spark-mln rdd labels batch-size]
      :as opts}]
  (cond (contains-many? opts :rdd :labels :batch-size)
        (.evaluate spark-mln rdd (into '() labels) batch-size)
        (contains-many? opts :rdd :labels)
        (.evaluate spark-mln rdd (into '() labels))
        (contains? opts :rdd)
        (.evaluate spark-mln rdd)))

(defn eval-regression-spark-mln
  "Evaluate the network in a distributed manner on the provided data

  :rdd (JavaRDD<org.nd4j.linalg.dataset.DataSet>), the data

  :mini-batch-size (int), Minibatch size to use when doing performing evaluation
   - optional"
  [& {:keys [spark-mln rdd mini-batch-size]}]
  (if mini-batch-size
    (.evaluateRegression spark-mln mini-batch-size)
    (.evaluateRegression spark-mln rdd)))

(defn eval-roc-spark-mln
  "Perform ROC analysis/evaluation on the given DataSet in a distributed manner

  :rdd (JavaRDD<org.nd4j.linalg.dataset.DataSet>) the data

  :threshold-steps (int), Number of threshold steps for ROC
   - see: dl4clj.eval.roc.rocs

  :mini-batch-size (int) Minibatch size to use when performing eval

  - :threshold-steps and :mini-batch-size are optional but must be supplied together"
  [& {:keys [spark-mln rdd threshold-steps mini-batch-size]
      :as opts}]
  (if (contains-many? opts :threshold-steps :mini-batch-size)
    (.evaluateROC spark-mln rdd threshold-steps mini-batch-size)
    (.evaluateROC spark-mln rdd)))

(defn eval-multi-class-roc-spark-mln
  "Perform ROC analysis/evaluation on the given DataSet in a distributed manner
   - using a ROCMultiClass evaluation object

  :rdd (JavaRDD<org.nd4j.linalg.dataset.DataSet>) the data

  :threshold-steps (int), Number of threshold steps for ROC
   - see: dl4clj.eval.roc.rocs

  :mini-batch-size (int) Minibatch size to use when performing eval

  - :threshold-steps and :mini-batch-size are optional but must be supplied together"
  [& {:keys [spark-mln rdd threshold-steps mini-batch-size]
      :as opts}]
  (if (contains-many? opts :threshold-steps :mini-batch-size)
    (.evaluateROCMultiClass spark-mln rdd threshold-steps mini-batch-size)
    (.evaluateROCMultiClass spark-mln rdd)))

(defn feed-forward-with-key
  "Feed-forward the specified data, with the given keys.

  :pair-rdd (JavaPairRDD<K,org.nd4j.linalg.api.ndarray.INDArray>) the data

  :batch-size (int), the batch size of the data"
  [& {:keys [spark-mln pair-rdd batch-size]}]
  (.feedForwardWithKey spark-mln pair-rdd batch-size))

(defn fit-spark-mln!
  "Fit the supplied model to the supplied dataset

  :rdd (JavaRDD<org.nd4j.linalg.dataset.DataSet>) the data
   - can also be RDD<org.nd4j.linalg.dataset.DataSet>

  :path-to-data (str), path to a directory of serialized DataSet objects
   - The assumption here is that the directory contains a number of DataSet objects,
     each serialized using (save-dataset! DataSet OutputStream)

  either :rdd or :path-to-data should be supplied, not both"
  [& {:keys [spark-mln rdd path-to-data n-epochs]
      :or {n-epochs 10}
      :as opts}]
  (if (contains? opts :rdd)
    (do (dotimes [n n-epochs]
          (if (contains? opts :rdd)
            (.fit spark-mln rdd)
            (.fit spark-mln path-to-data)))
        spark-mln)))

(defn fit-continous-labeled-point!
  "Fits a MultiLayerNetwork using Spark MLLib LabeledPoint instances
  This will convert labeled points that have continuous labels used
  for regression to the internal DL4J data format and train the model on that

  :rdd (JavaRDD<org.apache.spark.mllib.regression.LabeledPoint>) the data"
  [& {:keys [spark-mln rdd]}]
  (.fitContinuousLabeledPoint spark-mln rdd))

(defn fit-labeled-point!
  "Fit a MultiLayerNetwork using Spark MLLib LabeledPoint instances.

  :rdd JavaRDD<org.apache.spark.mllib.regression.LabeledPoint>"
  [& {:keys [spark-mln rdd]}]
  (.fitLabeledPoint spark-mln rdd))

(defn fit-from-paths!
  "Fit the network using a list of paths for serialized DataSet objects.

  :rdd JavaRDD<java.lang.String>"
  [& {:keys [spark-mln rdd]}]
  (.fitPaths spark-mln rdd))

(defn get-network
  "returns the mln from the Spark-mln"
  [spark-mln]
  (.getNetwork spark-mln))

(defn get-score
  "Gets the last (average) minibatch score from calling fit."
  [spark-mln]
  (.getScore spark-mln))

(defn get-spark-context
  "returns the spark context from the spark-mln"
  [spark-mln]
  (.getSparkContext spark-mln))

(defn get-spark-training-stats
  "Get the training statistics, after collection of stats has been enabled"
  [spark-mln]
  (.getSparkTrainingStats spark-mln))

(defn get-training-master
  "return the training master used to configure the spark-mln"
  [spark-mln]
  (.getTrainingMaster spark-mln))

(defn predict-spark-mln
  "feed data through the network and get its prediction

  :input (matrix or vector) the data to be fed through
   - org.apache.spark.mllib.linalg.Matrix or org.apache.spark.mllib.linalg.Vector"
  [& {:keys [spark-mln input]}]
  (.predict spark-mln input))

(defn score-examples-spark-mln
  "Score the examples individually

  :rdd (JavaRDD<org.nd4j.linalg.dataset.DataSet>) the data
   - can also be a pair-rdd (JavaPairRDD<K,org.nd4j.linalg.dataset.DataSet>) or
     RDD<org.nd4j.linalg.dataset.DataSet>

  :include-regularization-terms? (boolean), should regularization be factored into scoring

  :batch-size (int), the batch size for the data
   - optional arg"
  [& {:keys [spark-mln rdd include-regularization-terms? batch-size]}]
  (if (int? batch-size)
    (.scoreExamples spark-mln rdd include-regularization-terms? batch-size)
    (.scoreExamples spark-mln rdd include-regularization-terms?)))

;; to be removed

(defn set-network!
  "Set the network that underlies this SparkDl4jMultiLayer instacne
   - going to be removed in core branch
   - this is behind the scene method called in new-spark-multi-layer-network

  :mln (MultiLayerNetwork), the network to set
   - see: dl4clj.nn.multilayer.multi-layer-network

  returns the spark-mln"
  [& {:keys [spark-mln mln]}]
  (doto spark-mln (.setNetwork mln)))

(defn set-score-spark-mln!
  "sets the score for the supplied spark-mln
   - going to removed in core branch

  :score (double), the score to set

  This is an internal setter method but is being exposed
   - returs the spark-mln"
  [& {:keys [spark-mln score]}]
  (doto spark-mln (.setScore score)))
