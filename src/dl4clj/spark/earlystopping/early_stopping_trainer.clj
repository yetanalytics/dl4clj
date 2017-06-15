(ns dl4clj.spark.earlystopping.early-stopping-trainer
  (:import [org.deeplearning4j.spark.earlystopping SparkEarlyStoppingTrainer]))

(defn new-spark-early-stopping-trainer
  "object for conducting early stopping training via Spark with
   multi-layer networks

  :spark-context (spark), Can be either a JavaSparkContext or a SparkContext
   - need to make a ns dedicated to making JavaSparkContexts

  :training-master (training-master), the object which sets options for training on spark
   - see: TBD, need to write the ns which uses the training master builder class

  :early-stopping-conf (config), configuration for early stopping
   - see: dl4clj.earlystopping.early-stopping-config

  :mln (MultiLayerNetwork), the neural net to be trained
   - see: dl4clj.nn.multilayer.multi-layer-network for creating one from a nn-conf

  :training-rdd (JavaRdd <DataSet>), a dataset contained within a Java RDD.
   - the data for training

  :early-stopping-listener (listener), a listener which implements the EarlyStoppingListener interface
   - see: dl4clj.earlystopping.interfaces.listener
   - NOTE: building listeners in general still needs to be done"
  [& {:keys [spark-context training-master early-stopping-conf
             mln training-rdd early-stopping-listener]
      :as opts}]
  (if (contains? opts :early-stopping-listener)
    (SparkEarlyStoppingTrainer. spark-context training-master early-stopping-conf
                                mln training-rdd early-stopping-listener)
    (SparkEarlyStoppingTrainer. spark-context training-master early-stopping-conf
                                mln training-rdd)))

(defn fit-spark-es-trainer!
  "Fit the network used to create the early stopping trainer
   with the data supplied.

  :es-trainer (early stopping trainer), the object created by new-spark-early-stopping-trainer

  :rdd (JavaRdd <dataset or Multidataset>) the data to train on

  :multi-ds? (boolean), is the dataset within the RDD a multi-data-set?
   defaults to false, so expects rdd to contain a DataSet"
  [& {:keys [es-trainer rdd multi-ds?]
      :or {multi-ds? false}}]
  (if (true? multi-ds?)
    (doto es-trainer (.fitMulti rdd))
    (doto es-trainer (.fit rdd))))

(defn get-score-spark-es-trainer
  "returns the score of the model trained via spark"
  [fit-es-trainer]
  (.getScore fit-es-trainer))
