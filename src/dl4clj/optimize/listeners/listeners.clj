(ns ^{:doc "listener creation namespace.  composes the listeners package from dl4j
see: https://deeplearning4j.org/doc/org/deeplearning4j/optimize/listeners/package-summary.html"}
    dl4clj.optimize.listeners.listeners
  (:import [org.deeplearning4j.optimize.listeners
            ParamAndGradientIterationListener
            ComposableIterationListener
            ScoreIterationListener
            PerformanceListener
            PerformanceListener$Builder
            CollectScoresIterationListener])
  (:require [dl4clj.utils :refer [contains-many? generic-dispatching-fn]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; multi method that sets up the constructor/builder
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defmulti listeners generic-dispatching-fn)

(defmethod listeners :param-and-gradient [opts]
  (let [conf (:param-and-gradient opts)
        {iterations :iterations
         print-header? :print-header?
         print-mean? :print-mean?
         print-min-max? :print-min-max?
         print-mean-abs-value? :print-mean-abs-value?
         output-to-console? :output-to-console?
         output-to-file? :output-to-file?
         output-to-logger? :output-to-logger?
         file :file
         delim :delim} conf]
    (if (contains-many? conf :iterations :print-header? :print-mean?
                        :print-min-max? :print-mean-abs-value?
                        :output-to-console? :output-to-file? :output-to-logger?
                        :file :delim)
      (ParamAndGradientIterationListener. iterations print-header? print-mean?
                                          print-min-max? print-mean-abs-value?
                                          output-to-console? output-to-file?
                                          output-to-logger? file delim)
      (ParamAndGradientIterationListener.))))

(defmethod listeners :collection-scores [opts]
  (let [conf (:collection-scores opts)
        frequency (:frequency conf)]
    (if (nil? frequency)
      (CollectScoresIterationListener.)
      (CollectScoresIterationListener. frequency))))

(defmethod listeners :composable [opts]
  (let [conf (:composable opts)
        listeners (:listeners conf)]
    (ComposableIterationListener. listeners)))

(defmethod listeners :score-iteration [opts]
  (let [conf (:score-iteration opts)
        print-every-n (:print-every-n conf)]
    (if (nil? print-every-n)
      (ScoreIterationListener.)
      (ScoreIterationListener. print-every-n))))

(defmethod listeners :performance [opts]
  (let [conf (:performance opts)
        {report-batch? :report-batch?
         report-iteration? :report-iteration?
         report-sample? :report-sample?
         report-score? :report-score?
         report-time? :report-time?
         freq :frequency
         build? :build?} conf
        b (PerformanceListener$Builder.)]
    (cond-> b
      (contains? conf :report-batch?)
      (.reportBatch report-batch?)
      (contains? conf :report-iteration?)
      (.reportIteration report-iteration?)
      (contains? conf :report-sample?)
      (.reportSample report-sample?)
      (contains? conf :report-score?)
      (.reportScore report-score?)
      (contains? conf :report-time?)
      (.reportTime report-time?)
      (contains? conf :frequency)
      (.setFrequency freq)
      (true? build?)
      .build)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; user facing fns that specify args for making listeners
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-performance-iteration-listener
  "Simple IterationListener that tracks time spend on training per iteration.

  :report-batch (boolean), if batches/sec should be reported together with other data

  :report-iteration? (boolean), if iteration number should be reported together with other data

  :report-sample? (boolean), if samples/sec should be reported together with other data

  :report-score? (boolean),  if score should be reported together with other data

  :report-time? (boolean), if time per iteration should be reported together with other data

  :frequency (int), Desired IterationListener activation frequency

  :build? (boolean), if you want to build the builder
   - defaults to true"
  [& {:keys [report-batch? report-iteration? report-sample?
             report-score? report-time? build? frequency]
      :or {build? true
           report-batch? true
           report-iteration? true
           report-sample? true
           report-score? true
           report-time? true
           frequency 1}
      :as opts}]
  (listeners {:performance opts}))

(defn new-score-iteration-listener
  "Score iteration listener

  :print-every-n (int), print every n iterations"
  [& {:keys [print-every-n]
      :or {print-every-n 10}
      :as opts}]
  (listeners {:score-iteration opts}))

(defn new-composable-iteration-listener
  "A group of listeners

  listeners (collection or array) multiple listeners to compose together"
  [& {:keys [listeners]
      :as opts}]
  (listeners {:composable opts}))

(defn new-collection-scores-iteration-listener
  "CollectScoresIterationListener simply stores the model scores internally
  (along with the iteration) every 1 or N iterations (this is configurable).
  These scores can then be obtained or exported.

  :frequency (int), how often scores are stored"
  [& {:keys [frequency]
      :as opts}]
  (listeners {:collection-scores opts}))

(defn new-param-and-gradient-iteration-listener
  "An iteration listener that provides details on parameters and gradients at
  each iteration during traning. Attempts to provide much of the same information as
  the UI histogram iteration listener, but in a text-based format
  (for example, when learning on a system accessed via SSH etc).
  i.e., is intended to aid network tuning and debugging

  This iteration listener is set up to calculate mean, min, max, and
  mean absolute value of each type of parameter and gradient in the network
  at each iteration.

  :iterations (int), frequency to calculate and report values

  :print-header? (boolean), Whether to output a header row (i.e., names for each column)

  :print-mean? (boolean), Calculate and display the mean of parameters and gradients

  :print-min-max? (boolean), Calculate and display the min/max of the parameters and gradients

  :print-mean-abs-value? (boolean), Calculate and display the mean absolute value

  :output-to-console? (boolean), If true, display the values to the console

  :output-to-file? (boolean), If true, write the values to a file, one per line

  :output-to-logger? (boolean), If true, log the values

  :file (java.io.File), File to write values to. May be null
   - not used if :output-to-file? = false

  :delimiter (str), the delimiter for the output file."
  [& {:keys [iterations print-header? print-mean? print-min-max?
             print-mean-abs-value? output-to-console? output-to-file?
             output-to-logger? file delimiter]
      :or {iterations 1
           print-header? true
           print-mean? true
           print-min-max? true
           print-mean-abs-value? true
           output-to-console? true
           output-to-file? false
           output-to-logger? true
           file nil
           delimiter ","}
      :as opts}]
  (listeners {:param-and-gradient opts}))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Collection scores iteration listener specific fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn export-scores!
  "exports the scores from the collection scores listener
  to a file or output stream

  :file (java.io.File), a file to write to

  :delim (str), the delimiter for the file or output stream

  :output-stream (output stream), an output stream to write to

  returns the listener"
  [& {:keys [listener file delim output-stream]
      :as opts}]
  (cond (contains-many? opts :file :delim)
        (doto listener (.exportScores file delim))
        (contains-many? opts :output-stream :delim)
        (doto listener (.exportScores output-stream delim))
        (contains? opts :file)
        (doto listener (.exportScores file))
        (contains? opts :output-stream)
        (doto listener (.exportScores output-stream))
        :else
        (assert false "you must supply alteast a file or output stream to export to")))

(defn get-scores-vs-iter
  [listener]
  (.getScoreVsIter listener))
