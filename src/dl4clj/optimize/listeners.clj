(ns ^{:doc "listener creation namespace.  composes the listeners package from dl4j
see: https://deeplearning4j.org/doc/org/deeplearning4j/optimize/listeners/package-summary.html"}
    dl4clj.optimize.listeners
  (:import [org.deeplearning4j.optimize.listeners
            ParamAndGradientIterationListener
            ComposableIterationListener
            ScoreIterationListener
            PerformanceListener
            PerformanceListener$Builder
            CollectScoresIterationListener]
           [org.deeplearning4j.optimize.api IterationListener])
  (:require [dl4clj.utils :refer [contains-many? generic-dispatching-fn
                                  array-of]]))

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
      `(ParamAndGradientIterationListener. ~iterations ~print-header? ~print-mean?
                                          ~print-min-max? ~print-mean-abs-value?
                                          ~output-to-console? ~output-to-file?
                                          ~output-to-logger? ~file ~delim)
      `(ParamAndGradientIterationListener.))))

(defmethod listeners :collection-scores [opts]
  (let [conf (:collection-scores opts)
        frequency (:frequency conf)]
    (if frequency
      `(CollectScoresIterationListener. ~frequency)
      `(CollectScoresIterationListener.))))

(defmethod listeners :composable [opts]
  (let [conf (:composable opts)
        listeners (:listeners conf)]
    `(ComposableIterationListener. ~listeners)))

(defmethod listeners :score-iteration [opts]
  (let [conf (:score-iteration opts)
        print-every-n (:print-every-n conf)]
    (if print-every-n
      `(ScoreIterationListener. ~print-every-n)
      `(ScoreIterationListener.))))

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
    ;; update this to use builder fn
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
   - defaults to true

  :report-iteration? (boolean), if iteration number should be reported together with other data
   - defaults to true

  :report-sample? (boolean), if samples/sec should be reported together with other data
   - defaults to true

  :report-score? (boolean),  if score should be reported together with other data
   - defaults to true

  :report-time? (boolean), if time per iteration should be reported together with other data
   - defaults to true

  :frequency (int), Desired IterationListener activation frequency
   - defaults to 1

  :build? (boolean), if you want to build the builder
   - defaults to true

  :array? (boolean), if you want to return the object in an array of type IterationListener
   - defaults to false

  defaults are only used if no kw args are supplied"
  [& {:keys [report-batch? report-iteration? report-sample?
             report-score? report-time? build? frequency array?]
      :or {array? false}
      :as opts}]
  ;; refactor
  (let [conf (if (nil? opts)
               {:build? true
                :report-batch? true
                :report-iteration? true
                :report-sample? true
                :report-score? true
                :report-time? true
                :frequency 1}
               opts)]
    (if (true? array?)
    (array-of :data (listeners {:performance conf})
              :java-type IterationListener)
    (listeners {:performance conf}))))

(defn new-score-iteration-listener
  "Score iteration listener

  :print-every-n (int), print every n iterations
   - defaults to 10

  :array? (boolean), if you want to return the object in an array of type IterationListener
   - defaults to false

  :as-code? (boolean), return the java object or the code for creating it"
  [& {:keys [print-every-n array? as-code?]
      :or {array? false
           as-code? true}
      :as opts}]
  (let [conf (if opts
               opts
               {:print-every-n 10})
        code (if (true? array?)
               `(array-of :data ~(listeners {:score-iteration conf})
                         :java-type IterationListener)
               (listeners {:score-iteration conf}))]
    (if as-code?
      code
      (eval code))))

(defn new-composable-iteration-listener
  "A group of listeners

  listeners (collection or array) multiple listeners to compose together

  :array? (boolean), if you want to return the object in an array of type IterationListener
   - defaults to false

  :as-code? (boolean), return the java object or the code for creating it"
  [& {:keys [coll-of-listeners array? as-code?]
      :or {array? false
           as-code? true}
      :as opts}]
  (let [code (if (true? array?)
               `(array-of :data ~(listeners {:composable opts})
                         :java-type IterationListener)
               (listeners {:composable opts}))]
    (if as-code?
      code
      (eval code))))

(defn new-collection-scores-iteration-listener
  "CollectScoresIterationListener simply stores the model scores internally
  (along with the iteration) every 1 or N iterations (this is configurable).
  These scores can then be obtained or exported.

  :frequency (int), how often scores are stored
   - defaults to 1

  :array? (boolean), if you want to return the object in an array of type IterationListener
   - defaults to false

  :as-code? (boolean), return the java object or the code for creating it"
  [& {:keys [frequency array? as-code?]
      :or {array? false
           as-code? true}
      :as opts}]
  (let [conf (if opts
               opts
               {:frequency 1})
        code (if (true? array?)
               `(array-of :data ~(listeners {:collection-scores conf})
                         :java-type IterationListener)
               (listeners {:collection-scores conf}))]
    (if as-code?
      code
      (eval code))))

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
   - defaults to 1

  :print-header? (boolean), Whether to output a header row (i.e., names for each column)
   - defaults to true

  :print-mean? (boolean), Calculate and display the mean of parameters and gradients
   - defaults to true

  :print-min-max? (boolean), Calculate and display the min/max of the parameters and gradients
   - defaults to true

  :print-mean-abs-value? (boolean), Calculate and display the mean absolute value
   - defaults to true

  :output-to-console? (boolean), If true, display the values to the console
   - defaults to true

  :output-to-file? (boolean), If true, write the values to a file, one per line
   - defaults to false

  :output-to-logger? (boolean), If true, log the values
   - defaults to true

  :file (java.io.File), File to write values to. May be null
   - not used if :output-to-file? = false
   - defaults to nil

  :delimiter (str), the delimiter for the output file.
   - defaults to ,

  :array? (boolean), if you want to return the object in an array of type IterationListener
   - defaults to false

  :as-code? (boolean), return the java object or the code for creating it

  defaults are only used if no kw args are supplied"
  [& {:keys [iterations print-header? print-mean? print-min-max?
             print-mean-abs-value? output-to-console? output-to-file?
             output-to-logger? file delimiter array? as-code?]
      :or {array? false}
      :as opts}]
  (let [conf (if opts
               opts
               {:iterations 1
                :print-header? true
                :print-mean? true
                :print-min-max? true
                :print-mean-abs-value? true
                :output-to-console? true
                :output-to-file? false
                :output-to-logger? true
                :file nil
                :delimiter ","})
        code (if (true? array?)
               `(array-of :data ~(listeners {:param-and-gradient conf})
                         :java-type IterationListener)
               (listeners {:param-and-gradient conf}))]
    (if as-code?
      code
      (eval code))))
