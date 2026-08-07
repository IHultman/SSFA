import numpy as np
import sys


def ssfa_param_sweep(
  X: np.ndarray,
  rho_range: np.ndarray | None =None,
  n_rho: int =10,
  k_dims_max: np.ndarray | None =None,
  n_reps: int =1000,
  conv_thrsh: np.float32 =1e-9,
  adaptive_lasso: bool =False,
  verbose: bool =False):

  """
  Description:
    Estimate multiple sparse separable complex factor models over a grid of rho shrinkage
    parameters.

  Arguments:
    |
    |-------------- X: In the case of array-variate data, this is a (D + 1)-mode tensor of
    |                  dimension n x P1 x P2 x ... x PD, where n denotes the number of
    |                  observations, D denotes the number of modes of each array observation,
    |                  P1 is the dimension of the first mode, P2 is the dimension of the second
    |                  mode, so on and so forth. In the case of vector-variate data, this is an
    |                  n x P matrix.
    |
    |------ rho_range: OPTIONAL. If set, this must be a numpy vector with two positive values
    |                  representing the smallest and largest rho values respectively to use to
    |                  construct the grid of rho shrinkage parameters. If not set, a range of rho
    |                  values will be automatically chosen. It is recommended to let the rho
    |                  range be chosen automatically.
    |
    |---------- n_rho: OPTIONAL. Integer denoting the number of rho values in the rho shrinkage
    |                  parameter grid. An SSFA model will be fit at each value of rho. Default
    |                  is 10.
    |
    |----- k_dims_max: OPTIONAL. If set, this must be a numpy vector with D integers denoting
    |                  the maximum number of factors to estimate at each mode, where D denotes
    |                  the number of modes of each array observation in 'X'. As rho increases,
    |                  some of the columns of the factor loadings might zero out resulting in
    |                  fewer factors estimated for the corresponding model.
    |
    |--------- n_reps: OPTIONAL. Integer denoting the max number of iterations to run the EM
    |                  algorithm for each model fit. Default is 1000;
    |
    |----- conv_thrsh: OPTIONAL. Floating point value used to determine convergence of the EM
    |                  algorithm. Default is 1e-9.
    |
    |- adaptive_lasso: OPTIONAL. Boolean indicating whether or not to use adaptive lasso when
    |                  estimating factor loadings. If True, then a warmup model is fit at rho
    |                  equals 0.0 and the estimated factor loadings are used to set the adaptive
    |                  lasso weights. Default is False which amounts to setting the weights to
    |                  all be equal at 1.
    |
    |-------- verbose: OPTIONAL. Boolean indicating whether or not to print algorithm runtime
    |                  details. Default is False.

  Return values:
    |
    |---------- model: List of the SSFA models fit at each value of rho. Each model is a
    |                  python dictionary with fields described in the 'ssfa()' function
    |                  documentation below.
    |
    |- model_min_ebic: The SSFA model with the smallest extended BIC (EBIC) metric.
    |
    |- model_1se_ebic: The SSFA model with the most sparse factors whose EBIC is within 1
    |                  median absolute deviation of the smallest EBIC.
    |
    |-------- ix_best: The index of the model in the 'model' list with the smallest EBIC.
    |
    |--------- ix_1se: The index of the model in the 'model' list with the mose sparse factors
    |                  whose EBIC is within 1 median absolute deviation of the smallest EBIC.
    |
    |------------ rho: The vector of rho values used to fit each model in the 'model' list.
    |
  """

  if not isinstance(X, np.ndarray):
    err_msg = "The provided variable 'X' must be a numpy array.";
    raise TypeError(err_msg);

  n_modes = len(X.shape);

  if n_modes < 2:
    err_msg = "The provided variable 'X' must be either a numpy matrix or array.";
    raise AssertionError(err_msg);

  n_dim = X.shape[0];
  p_dims = np.array(X.shape[1:]);
  d_dim = len(p_dims);

  if rho_range is not None:
    if not isinstance(rho_range, np.ndarray):
      err_msg = "The provided variable 'rho_range' must be a numpy vector.";
      raise TypeError(err_msg);

    if len(rho_range) != 2:
      err_msg = "The length of the provided vector 'rho_range' must equal two.";
      raise AssertionError(err_msg);

    if not np.all((rho_range > 0) & (rho_range < np.inf) ):
      err_msg = "The elements of 'rho_range' must be positive.";
      raise AssertionError(err_msg);

  if not ((n_rho > 0) and ((n_rho % 1) == 0) ):
    err_msg = "The provided variable 'n_rho' must be a positive integer."
    raise AssertionError(err_msg);

  if k_dims_max is None:
    k_dims_max = np.array(np.floor(np.log(p_dims) + 1), dtype=int);

  if (not isinstance(k_dims_max, np.ndarray) ) or (not np.all(((k_dims_max % 1) == 0) )):
    err_msg = "The provided variable 'k_dims_max' must be a numpy vector of integers.";
    raise TypeError(err_msg);

  if len(k_dims_max) != d_dim:
    err_msg = \
      "The length of the provided vector 'k_dims_max' must equal the dimensions\n" +\
      "of 'X' minus one.";

    raise AssertionError(err_msg);

  for jx in range(d_dim):
    ixs_no_jx = np.array(sorted(set(range(d_dim) ).difference({jx}) ));

    if d_dim > 1:
      n_no_jx = n_dim * np.prod(p_dims[ixs_no_jx]);
    else:
      n_no_jx = n_dim;

    k_dim_upper_bound = np.min((p_dims[jx], n_no_jx) );

    if k_dims_max[jx] >= k_dim_upper_bound:
      k_dims_max[jx] = k_dim_upper_bound - 1;

      if d_dim > 1:
        id_mode = jx + 1;
        warn_msg = \
          "The requested upper bound on the number of factors for the mode-{} covariance\n" +\
          "matrix should be less than the min(# of mode-{} fibres, len(mode-{} fibres) ) = {}.\n" +\
          "Resetting max number of mode-{} factors to {}.";

        warn_msg = warn_msg.format(
          id_mode, id_mode, id_mode, k_dim_upper_bound, id_mode, k_dim_upper_bound - 1);

      else:
        warn_msg = \
          "The requested upper bound on the number of factors should be less than the\n" +\
          "min(# of observations, # of variables) = {}. Resetting max number of factors\n" +\
          "to {}.";

        warn_msg = warn_msg.format(k_dim_upper_bound, k_dim_upper_bound - 1);

      print(warn_msg);
      sys.stdout.flush();

  if not np.all(k_dims_max > 0):
    err_msg = "The elements of 'k_dims_max' must be positive integers."
    raise AssertionError(err_msg);

  if not ((n_reps > 0) and ((n_reps % 1) == 0) ):
    err_msg = "The provided variable 'n_reps' must be a positive integer."
    raise AssertionError(err_msg);

  # Initialize covariance paramters.
  Psi_init = [np.random.uniform(1, 2, p_dims[jx]) for jx in range(d_dim)];
  Lam_init = [rcnorm(p_dims[jx] * k_dims_max[jx]).reshape((p_dims[jx], -1) ) for jx in range(d_dim)];

  Lam_mats = [Lam_jx.copy() for Lam_jx in Lam_init];
  Psi_diags = [Psi_jx.copy() for Psi_jx in Psi_init];

  adapt_penalty = [np.ones(Lam_jx.shape) for Lam_jx in Lam_init];

  try:
    ssfa_warmup = ssfa(
      X,
      None,
      Lam_mats,
      Psi_diags,
      adapt_penalty,
      0.0,
      n_reps,
      conv_thrsh,
      verbose);
  except SSFA_RuntimeError:
    ssfa_warmup = None;

  if ssfa_warmup is not None:
    for jx in range(d_dim):
      Psi_diags[jx][:] = ssfa_warmup["Psi"][jx];
      Lam_mats[jx][:] = ssfa_warmup["Lam"][jx];

  if rho_range is None:
    if ssfa_warmup is None:
      err_msg = \
        "Cannot estimate rho range from warm-up model. Either manually set rho range or\n" +\
        "reduce the maximum number of factors to consider.";

      raise Exception(err_msg);

    # Compute the diagonal components of the Psi matrices under the setting where the factor
    # loadings are set to zero. In this case the Psi diagonals should be at their largest
    # which results in the largest shrinkage. In this way we can work backwards and determine
    # what the maximum value of rho needs to be in order to zero out all of the factor loadings.
    Psi_max_diags = get_Psi_max(X);

    rho_min = [];
    rho_max = [];

    for jx in range(d_dim):
      k_jx = Lam_mats[jx].shape[1];
      Psi_max_inv_Lam = (
        (1 / Psi_max_diags[jx]).reshape((-1, 1) ).repeat(k_jx, 1) *
        np.abs(Lam_mats[jx]) );

      rho_min.append(np.min(2 * Psi_max_inv_Lam) );
      rho_max.append(np.max(2 * Psi_max_inv_Lam) );

    rho_range = np.array([np.min(rho_min) / 10, np.min(rho_max)]);

  else:
    rho_max = rho_range.max();
    rho_range[0] = rho_range.min();
    rho_range[1] = rho_max;

  rho_vals = np.logspace(np.log10(rho_range[0]), np.log10(rho_range[1]), n_rho);

  if adaptive_lasso:
    if ssfa_warmup is None:
      err_msg = \
        "Cannot initialize adaptive lasso weights from warm-up model. Try reducing\n" +\
        "the maximum number of factors to consider.";

      raise Exception(err_msg);

    for jx in range(d_dim):
      pen_jx = 1 / np.abs(ssfa_warmup["Lam"][jx]);
      ixs_inf = np.isinf(pen_jx);
      pen_jx[ixs_inf] = pen_jx[~ixs_inf].max();
      adapt_penalty[jx][:] = pen_jx;

  ssfa_models = [];
  rho_vals_final = [];

  if ssfa_warmup is not None:
    ssfa_models.append(ssfa_warmup);
    rho_vals_final.append(0.0);

  for rho_ix in rho_vals:
    try:
      ssfa_ix = ssfa(
        X,
        None,
        Lam_mats,
        Psi_diags,
        adapt_penalty,
        rho_ix,
        n_reps,
        conv_thrsh,
        verbose);
    except SSFA_RuntimeError:
      ssfa_ix = None;

    if ssfa_ix is not None:
      ssfa_models.append(ssfa_ix);
      rho_vals_final.append(rho_ix);

      for jx in range(d_dim):
        Psi_diags[jx][:] = ssfa_ix["Psi"][jx];
        Lam_mats[jx][:] = ssfa_ix["Lam"][jx];

    else:
      for jx in range(d_dim):
        Psi_diags[jx][:] = Psi_init[jx];
        Lam_mats[jx][:] = Lam_init[jx];

  rho_vals_final = np.array(rho_vals_final);

  ixs_all_zero = np.array([
    np.all([np.all(np.abs(ssfa_models[ix]["Lam"][jx]) == 0) for jx in range(d_dim)])
    for ix in range(len(ssfa_models) )]);

  max_ix_nz_model = len(ssfa_models);

  if np.any(ixs_all_zero):
    max_ix_nz_model = np.min(np.flatnonzero(ixs_all_zero) ) + 1;

  ssfa_models = ssfa_models[0:max_ix_nz_model];
  rho_vals_final = rho_vals_final[0:max_ix_nz_model];

  ebic_vals = np.array([ssfa_models[ix]["ebic"] for ix in range(len(ssfa_models) )]);
  mad_ebic = np.median(np.abs(ebic_vals - np.median(ebic_vals) ));
  ix_best = np.max(np.flatnonzero(ebic_vals == np.min(ebic_vals) ));
  upper_ebic = ebic_vals[ix_best] + mad_ebic;
  ix_1se = np.max(np.flatnonzero(ebic_vals < upper_ebic) );

  return {
    "model": ssfa_models,
    "model_min_ebic": ssfa_models[ix_best],
    "model_1se_ebic": ssfa_models[ix_1se],
    "ix_best": ix_best,
    "ix_1se": ix_1se,
    "rho": rho_vals_final};


def ssfa_vector(
  X: np.ndarray,
  k_dim: np.ndarray | int =None,
  Lam_init: np.ndarray | list | None =None,
  Psi_init: np.ndarray | list | None =None,
  adapt_penalty: np.ndarray | list | None =None,
  rho: np.float32 =0.0,
  n_reps: int =1000,
  conv_thrsh: np.float32 =1e-9,
  verbose: bool =False):

  """
  Description:
    Compute the sparse complex factor model for vector-variate data.

  Arguments:
    |
    |------------- X: Matrix of dimension n x P, where n denotes the number of observations,
    |                 and P is the dimension of each vector observation. If the observations
    |                 in 'X' are array-variate, use the 'ssfa_array()' function instead.
    |
    |--------- k_dim: OPTIONAL. If set, this must be either an integer or a numpy vector with
    |                 a single integer denoting the number of factors to estimate.
    |
    |------ Lam_init: OPTIONAL. Initial factor loadings matrix. If set, this must be either
    |                 a numpy matrix or a list with a single numpy matrix. The number of
    |                 columns of the matrix will determine the number of factors estimated.
    |                 This argument will override the 'k_dim' argument.
    |
    |------ Psi_init: OPTIONAL. Initial diagonal terms of the unique variance matrix. If set,
    |                 this must be either a P-dimensional numpy vector or a list with a single
    |                 P-dimensional numpy vector where P denotes the dimension of the vector
    |                 observations provided in the 'X' matrix.
    |
    |- adapt_penalty: OPTIONAL. Adaptive lasso penalty weights for factor loadings matrix. If
    |                 set, this must be either a numpy matrix or a list with a single numpy
    |                 matrix. If 'Lam_init' is not set, the number of columns of this matrix
    |                 will determine the number of factors estimated. If 'Lam_init' is set,
    |                 the dimensions of this matrix must equal the dimensions of the matrix
    |                 in 'Lam_init'. This argument will override the 'k_dim' argument.
    |
    |----------- rho: OPTIONAL. Floating point sparsity shrinkage parameter. Default is 0.0;
    |
    |-------- n_reps: OPTIONAL. Integer denoting the max number of iterations to run the EM
    |                 algorithm. Default is 1000;
    |
    |---- conv_thrsh: OPTIONAL. Floating point value used to determine convergence of the EM
    |                 algorithm. Default is 1e-9.
    |
    |------- verbose: OPTIONAL. Boolean indicating whether or not to print algorithm runtime
    |                 details. Default is False.

  Return values:
    |
    |------- Lam: List with a single factor loadings matrix.
    |
    |------- Psi: List with a single P-dimensional vector providing the unique variances of
    |             the observation variables.
    |
    |-------- ll: The log-likelihood under the SSFA model based on the final model
    |             estimates.
    |
    |------ ebic: The extended BIC metric based on the final model estimates and the
    |             log-likelihood.
    |
    |- norm_diff: A vector providing the relative norm of the difference between one
    |             iteration's estimate of the covariance structure and the previous
    |             iteration's estimate of the covariance structure.
  """

  if not isinstance(X, np.ndarray):
    err_msg = "The provided variable 'X' must be a numpy matrix.";
    raise TypeError(err_msg);

  n_modes = len(X.shape);

  if n_modes > 2:
    err_msg = "The provided array 'X' must be a matrix.";
    raise AssertionError(err_msg);

  n_dim, p_dim = X.shape;

  if Lam_init is not None:
    if isinstance(Lam_init, list):
      if (len(Lam_init) != 1) or not isinstance(Lam_init[0], np.ndarray):
        err_msg = "The provided list 'Lam_init' should contain exactly one numpy matrix.";
        raise TypeError(err_msg);

      Lam_init = Lam_init[0];

    if not isinstance(Lam_init, np.ndarray):
      err_msg = "The provided variable 'Lam_init' must be a numpy matrix.";
      raise TypeError(err_msg);

    p_dim_check, k_dim = Lam_init.shape;

    if not (p_dim_check == p_dim):
      err_msg = \
        "The matrix 'Lam_init' must have a number of rows equal to the number of columns of X.";

      raise AssertionError(err_msg);

    if not ((k_dim > 0) and (k_dim < p_dim) ):
      err_msg = \
        "The matrix 'Lam_init' must have a number of columns less than its number of rows.";

      raise AssertionError(err_msg);

  if Psi_init is not None:
    if isinstance(Psi_init, list):
      if (len(Psi_init) != 1) or not isinstance(Psi_init[0], np.ndarray):
        err_msg = "The provided list 'Psi_init' should contain exactly one numpy vector.";
        raise TypeError(err_msg);

      Psi_init = Psi_init[0];

    if not isinstance(Psi_init, np.ndarray):
      err_msg = "The provided variable 'Psi_init' must be a numpy vector.";
      raise TypeError(err_msg);

    p_dim_check = len(Psi_init);

    if not (p_dim_check == p_dim):
      err_msg = \
        "The vector 'Psi_init' must have a number of elements equal to the number of\n" +\
        "columns of X.";

      raise AssertionError(err_msg);

  if adapt_penalty is not None:
    if isinstance(adapt_penalty, list):
      if (len(adapt_penalty) != 1) or not isinstance(adapt_penalty[0], np.ndarray):
        err_msg = "The provided list 'adapt_penalty' should contain exactly one numpy matrix.";
        raise TypeError(err_msg);

      adapt_penalty = adapt_penalty[0];

    if not isinstance(adapt_penalty, np.ndarray):
      err_msg = "The provided variable 'adapt_penalty' must be a numpy matrix.";
      raise TypeError(err_msg);

    p_dim_check, k_dim_check = adapt_penalty.shape;

    if Lam_init is not None:
      if not ((p_dim_check == p_dim) and (k_dim_check == k_dim) ):
        err_msg = "The matrix 'adapt_penalty' must have the same dimensions as 'Lam_init'.";
        raise AssertionError(err_msg);

    else:
      k_dim = k_dim_check;

      if not (p_dim_check == p_dim):
        err_msg = \
          "The matrix 'adapt_penalty' must have a number of rows equal to the number of columns of X.";

        raise AssertionError(err_msg);

      if not ((k_dim > 0) and (k_dim < p_dim) ):
        err_msg = \
          "The matrix 'adapt_penalty' must have a number of columns less than its number of rows.";

        raise AssertionError(err_msg);

  if k_dim is None:
    k_dim = int(np.floor(np.log(p_dim) + 1) );

  else:
    if isinstance(k_dim, np.ndarray):
      if len(k_dim) != 1:
        err_msg = "The provided vector 'k_dim' should contain exactly one integer.";
        raise TypeError(err_msg);

      k_dim = k_dim[0];

    if not ((k_dim % 1) == 0):
      err_msg = "The provided argument 'k_dim' must contain an integer.";
      raise TypeError(err_msg);

    if not ((k_dim > 0) and (k_dim < p_dim) ):
      err_msg = \
        "The provided argument 'k_dim' must contain a positive integer less than the\n" +\
        "dimension of the vector obervations in 'X'.";

      raise AssertionError(err_msg);

  if isinstance(rho, int):
    rho = np.float32(rho);

  if rho < 0.0:
    err_msg = "The sparsity parameter rho must be non-negative.";
    raise AssertionError(err_msg);

  if not ((n_reps > 0) and ((n_reps % 1) == 0) ):
    err_msg = "The provided variable 'n_reps' must be a positive integer."
    raise AssertionError(err_msg);

  if Lam_init is None:
    Lam_init = rcnorm(p_dim * k_dim).reshape((p_dim, -1) );

  if Psi_init is None:
    Psi_init = np.random.uniform(1, 2, p_dim);

  if adapt_penalty is None:
    adapt_penalty = np.ones(Lam_init.shape);

  Lam_mat = Lam_init.copy();
  Psi_diag = Psi_init.copy();

  ixs_diag = np.arange(p_dim);

  Sig_mat_prev = Lam_init @ Lam_init.conj().T;
  Sig_mat_prev[ixs_diag,ixs_diag] = Sig_mat_prev[ixs_diag,ixs_diag] + Psi_diag;

  Sig_mat = Sig_mat_prev.copy();

  if (n_dim > p_dim):
    R = np.linalg.qr(X.conj(), 'r');
    XtX = R.conj().T @ R;

  else:
    XtX = X.T @ X.conj();

  norm_diff_vals = np.array([np.nan for ix in range(n_reps)]);

  for rx in range(n_reps):
    if verbose:
      if rx % 10 == 0:
        print("Rep {} ...".format(rx) );
        sys.stdout.flush();

    Lam_new, Psi_new = pxem_w_lasso(
      XtX,
      n_dim,
      Lam_mat,
      Psi_diag,
      adapt_penalty,
      rho);

    early_terminate = (
      np.any(np.isnan(Lam_new) ) or
      np.any(Psi_new == 0) or
      np.any(np.isnan(Psi_new) ));

    if early_terminate:
      raise SSFA_RuntimeError;

    Lam_mat[:] = Lam_new;
    Psi_diag[:] = Psi_new;
    Sig_mat[:] = Lam_new @ Lam_new.conj().T;
    Sig_mat[ixs_diag,ixs_diag] = Sig_mat[ixs_diag,ixs_diag] + Psi_new;

    norm_diff_vals[rx] = np.exp(
      np.log(np.linalg.norm(Sig_mat - Sig_mat_prev, "fro") ) -
      np.log(np.linalg.norm(Sig_mat, "fro") ));

    Sig_mat_prev[:] = Sig_mat;

    break_condition = (
      np.all(Lam_mat == 0) or
      (norm_diff_vals[rx] < conv_thrsh and rx > 10) );

    if break_condition:
      break;

  try:
    ll = calculate_likelihood_vector(X, Lam_mat, Psi_diag);
  except:
    raise SSFA_RuntimeError;

  n_tot_obs = np.prod(X.shape);
  n_nonzero = np.sum(np.abs(Lam_mat) > 0) + p_dim;
  n_tot_models = p_dim * (k_dim + 1);
  log_n_choose_k = (
    log_n_factorial(n_tot_models) -
    log_n_factorial(n_tot_models - n_nonzero) -
    log_n_factorial(n_nonzero) );

  ebic = -2 * ll + n_nonzero * np.log(n_tot_obs) + 2 * log_n_choose_k;

  return {
    "Lam": [Lam_mat],
    "Psi": [Psi_diag],
    "ll": ll,
    "ebic": ebic,
    "norm_diff": norm_diff_vals[0:(rx + 1)]};


def ssfa_array(
  X: np.ndarray,
  k_dims: np.ndarray | None =None,
  Lam_init: list | None =None,
  Psi_init: list | None =None,
  adapt_penalty: list | None =None,
  rho: np.float32 =0.0,
  n_reps: int =1000,
  conv_thrsh: np.float32 =1e-9,
  verbose: bool =False):

  """
  Description:
    Compute the sparse separable complex factor model for array-variate data.

  Arguments:
    |
    |------------- X: (D + 1)-mode tensor of dimension n x P1 x P2 x ... x PD, where n denotes
    |                 the number of observations, D denotes the number of modes of each array
    |                 observation, P1 is the dimension of the first mode, P2 is the dimension
    |                 of the second mode, so on and so forth. I.e. this is an array of n
    |                 D-dimensional subarrays. The number of modes, D, must be greater than 1.
    |                 If D equals 1, use the 'ssfa_vector()' function instead.
    |
    |-------- k_dims: OPTIONAL. If set, this must be a numpy vector with D integers denoting
    |                 the number of factors to estimate at each mode, where D denotes the
    |                 number of modes of each array observation in 'X'.
    |
    |------ Lam_init: OPTIONAL. Initial factor loadings matrices. If set, this must be a list
    |                 with D elements, each of which must be a numpy matrix. The number of
    |                 columns of each matrix in this list will determine the number of factors
    |                 estimated for each mode. This argument will override the 'k_dims' argument.
    |
    |------ Psi_init: OPTIONAL. Initial diagonal terms of unique variance matrices. If set, this
    |                 must be a list with D elements, each of which must be a Pj-dimensional
    |                 numpy vector where Pj denotes the dimension of the jth mode of the array
    |                 observations in 'X'.
    |
    |- adapt_penalty: OPTIONAL. Adaptive lasso penalty weights for factor loadings matrices.
    |                 If set, this must be a list with D elements, each of which must be a
    |                 numpy matrix. If 'Lam_init' is not set, the number of columns of each
    |                 matrix in this list will determine the number of factors estimated for
    |                 each mode. If 'Lam_init' is set, the dimensions of each matrix must equal
    |                 the dimensions of the corresponding matrices in 'Lam_init'. This will
    |                 argument override the 'k_dims' argument.
    |
    |----------- rho: OPTIONAL. Floating point sparsity shrinkage parameter. Default is 0.0;
    |
    |-------- n_reps: OPTIONAL. Integer denoting the max number of iterations to run the EM
    |                 algorithm. Default is 1000;
    |
    |---- conv_thrsh: OPTIONAL. Floating point value used to determine convergence of the EM
    |                 algorithm. Default is 1e-9.
    |
    |------- verbose: OPTIONAL. Boolean indicating whether or not to print algorithm runtime
    |                 details. Default is False.

  Return values:
    |
    |----------- Lam: List of D factor loadings matrices corresponding to each mode of the
    |                 data.
    |
    |----------- Psi: List of D numpy vectors providing the unique variances corresponding to
    |                 each mode of the data.
    |
    |------------ ll: The log-likelihood under the SSFA model based on the final model
    |                 estimates.
    |
    |---------- ebic: The extended BIC metric based on the final model estimates and the
    |                 log-likelihood.
    |
    |- max_norm_diff: A vector providing the max relative norm of the difference between one
    |                 iteration's estimates of each mode's covariance structures and the
    |                 previous iteration's estimates of each mode's covariance structures.
  """

  if not isinstance(X, np.ndarray):
    err_msg = "The provided variable 'X' must be a numpy array.";
    raise TypeError(err_msg);

  n_modes = len(X.shape);

  if n_modes < 3:
    err_msg = "The provided array 'X' must have dimensions greater than two.";
    raise AssertionError(err_msg);

  n_dim = X.shape[0];
  p_dims = np.array(X.shape[1:]);
  d_dim = len(p_dims);

  if Lam_init is not None:
    if not isinstance(Lam_init, list):
      err_msg = "The provided variable 'Lam_init' must be a list of numpy matrices.";
      raise TypeError(err_msg);

    type_check = np.array([isinstance(Lam_jx, np.ndarray) for Lam_jx in Lam_init]);

    if not np.all(type_check):
      err_msg = "The provided variable 'Lam_init' must be a list of numpy matrices.";
      raise TypeError(err_msg);

    if len(Lam_init) != d_dim:
      err_msg = \
        "The length of the provided list 'Lam_init' must equal the dimensions of\n" +\
        "'X' minus one.";

      raise AssertionError(err_msg);

    p_dims_check = np.array([Lam_jx.shape[0] for Lam_jx in Lam_init]);
    k_dims = np.array([Lam_jx.shape[1] for Lam_jx in Lam_init]);

    if not np.all(p_dims_check == p_dims):
      err_msg = \
        "The matrices of 'Lam_init' must each have a number of rows equal to the corresponding\n" +\
        "dimension of the 'X' array.";

      raise AssertionError(err_msg);

    if not np.all((k_dims > 0) & (k_dims < p_dims) ):
      err_msg = \
        "The matrices of 'Lam_init' must each have a number of columns less than\n" +\
        "its number of rows.";

      raise AssertionError(err_msg);

  if Psi_init is not None:
    if not isinstance(Psi_init, list):
      err_msg = "The provided variable 'Psi_init' must be a list of numpy vectors.";
      raise TypeError(err_msg);

    type_check = np.array([isinstance(Psi_jx, np.ndarray) for Psi_jx in Psi_init]);

    if not np.all(type_check):
      err_msg = "The provided variable 'Psi_init' must be a list of numpy vectors.";
      raise TypeError(err_msg);

    if len(Psi_init) != d_dim:
      err_msg = "The length of the provided list 'Psi_init' must equal the dimensions of 'X' minus one.";
      raise AssertionError(err_msg);

    vector_check = np.array([(len(Psi_jx.shape) == 1) for Psi_jx in Psi_init]);

    if not np.all(vector_check):
      err_msg = "The provided variable 'Psi_init' must be a list of numpy vectors.";
      raise TypeError(err_msg);

    p_dims_check = np.array([len(Psi_jx) for Psi_jx in Psi_init]);

    if not np.all(p_dims_check == p_dims):
      err_msg = \
        "The vectors of 'Psi_init' must each have a number of elements equal to the corresponding\n" +\
        "dimension of the 'X' array observations.";

      raise AssertionError(err_msg);

  if adapt_penalty is not None:
    if not isinstance(adapt_penalty, list):
      err_msg = "The provided variable 'adapt_penalty' must be a list of numpy matrices.";
      raise TypeError(err_msg);

    type_check = np.array([isinstance(mat_jx, np.ndarray) for mat_jx in adapt_penalty]);

    if not np.all(type_check):
      err_msg = "The provided variable 'adapt_penalty' must be a list of numpy matrices.";
      raise TypeError(err_msg);

    if len(adapt_penalty) != d_dim:
      err_msg = "The length of the provided list 'adapt_penalty' must equal the dimensions of 'X' minus one.";
      raise AssertionError(err_msg);

    p_dims_check = np.array([mat_jx.shape[0] for mat_jx in adapt_penalty]);
    k_dims_check = np.array([mat_jx.shape[1] for mat_jx in adapt_penalty]);

    if Lam_init is not None:
      if not np.all((p_dims_check == p_dims) & (k_dims_check == k_dims) ):
        err_msg = \
          "The matrices of 'adapt_penalty' must each have dimensions equal to the corresponding " +\
          "matrix in 'Lam_init'.";

        raise AssertionError(err_msg);

    else:
      k_dims = k_dims_check;

      if not np.all(p_dims_check == p_dims):
        err_msg = \
          "The matrices of 'adapt_penalty' must each have a number of rows equal to the corresponding\n" +\
          "dimension of the 'X' array.";

        raise AssertionError(err_msg);

      if not np.all((k_dims > 0) & (k_dims < p_dims) ):
        err_msg = \
          "The matrices of 'adapt_penalty' must each have a number of columns less than\n" +\
          "its number of rows.";

        raise AssertionError(err_msg);

  if k_dims is None:
    k_dims = np.array(np.floor(np.log(p_dims) + 1), dtype=int);

  else:
    if (not isinstance(k_dims, np.ndarray) ) or (not np.all(((k_dims % 1) == 0) )):
      err_msg = "The provided variable 'k_dims' must be a numpy vector of integers.";
      raise TypeError(err_msg);

    if len(k_dims) != d_dim:
      err_msg = \
        "The length of the provided vector 'k_dims' must equal the dimensions\n" +\
        "of 'X' minus one.";

      raise AssertionError(err_msg);

    if not np.all((k_dims > 0) & (k_dims < p_dims) ):
      err_msg = \
        "The elements of 'k_dims' must be positive integers less than the corresponding\n" +\
        "dimensions of the array obervations in 'X'.";

      raise AssertionError(err_msg);

  if isinstance(rho, int):
    rho = np.float32(rho);

  if rho < 0.0:
    err_msg = "The sparsity parameter rho must be non-negative.";
    raise AssertionError(err_msg);

  if not ((n_reps > 0) and ((n_reps % 1) == 0) ):
    err_msg = "The provided variable 'n_reps' must be a positive integer."
    raise AssertionError(err_msg);

  if Lam_init is None:
    Lam_init = [rcnorm(p_dims[jx] * k_dims[jx]).reshape((p_dims[jx], -1) ) for jx in range(d_dim)];

  if Psi_init is None:
    Psi_init = [np.random.uniform(1, 2, p_dims[jx]) for jx in range(d_dim)];

  if adapt_penalty is None:
    adapt_penalty = [np.ones(Lam_jx.shape) for Lam_jx in Lam_init];

  Lam_mats = [Lam_jx.copy() for Lam_jx in Lam_init];
  Psi_diags = [Psi_jx.copy() for Psi_jx in Psi_init];

  ixs_diags = [np.arange(p_dim) for p_dim in p_dims];

  Sig_mats_prev = [
    Lam_init[jx] @ Lam_init[jx].conj().T
    for jx in range(d_dim)];

  for jx in range(d_dim):
    Sig_mats_prev[jx][ixs_diags[jx],ixs_diags[jx]] = (
      Sig_mats_prev[jx][ixs_diags[jx],ixs_diags[jx]] + Psi_diags[jx]);

  log_max_diags = np.array([
    np.log(np.min(Psi_diags[jx]) )
    for jx in range(d_dim)]);

  scale_factrs = np.exp(np.mean(log_max_diags) - log_max_diags);

  for jx in range(d_dim):
    Lam_mats[jx][:] = np.sqrt(scale_factrs[jx]) * Lam_mats[jx];
    Psi_diags[jx][:] = scale_factrs[jx] * Psi_diags[jx];
    Sig_mats_prev[jx][:] = scale_factrs[jx] * Sig_mats_prev[jx];

  Sig_mats = [Sig_mats_prev[jx].copy() for jx in range(d_dim)];

  ixs_transpose_left = [*list(range(1, n_modes) ), 0];

  max_norm_diff_vals = np.empty(n_reps);
  max_norm_diff_vals[:] = np.nan;

  for rx in range(n_reps):
    if verbose:
      if rx % 10 == 0:
        print("Rep {} ...".format(rx) );
        sys.stdout.flush();

    for jx in range(d_dim):
      #print("jx = " + str(jx) );
      #print("  p_jx = " + str(p_dims[jx]) );

      L_inv_mats = [];
      ixs_no_jx = np.array(sorted(set(range(d_dim) ).difference({jx}) ));

      #print("  not jx = " + str(ixs_no_jx) );

      for njx in ixs_no_jx:
        Sig_njx = Lam_mats[njx] @ Lam_mats[njx].conj().T;
        Sig_njx[ixs_diags[njx],ixs_diags[njx]] = Sig_njx[ixs_diags[njx],ixs_diags[njx]] + Psi_diags[njx];
        L_inv_mats.append(np.linalg.inv(np.linalg.cholesky(Sig_njx) ));

      ixs_transpose = np.array([*(ixs_no_jx + 1), jx + 1, 0]);

      #print("  ixs_transpose = " + str(ixs_transpose) );

      X_jx = tucker_prod_seq(X.transpose(ixs_transpose), L_inv_mats).transpose(ixs_transpose_left);

      #print("  X_jx.shape = " + str(X_jx.shape) );

      X_jx = X_jx.reshape((X_jx.shape[0], -1) );
      n_no_jx = X_jx.shape[1];

      if (X_jx.shape[1] >= X_jx.shape[0]):
        R_jx = np.linalg.qr(X_jx.conj().T, 'r');
        XXt = R_jx.conj().T @ R_jx;
      else:
        XXt = X_jx @ X_jx.conj().T;

      Lam_new, Psi_new = pxem_w_lasso(
        XXt,
        n_no_jx,
        Lam_mats[jx],
        Psi_diags[jx],
        adapt_penalty[jx],
        rho);

      early_terminate = (
        np.any(np.isnan(Lam_new) ) or
        np.any(Psi_new == 0) or
        np.any(np.isnan(Psi_new) ));

      if early_terminate:
        raise SSFA_RuntimeError;

      Lam_mats[jx][:] = Lam_new;
      Psi_diags[jx][:] = Psi_new;
      Sig_mats[jx][:] = Lam_new @ Lam_new.conj().T;
      Sig_mats[jx][ixs_diags[jx],ixs_diags[jx]] = Sig_mats[jx][ixs_diags[jx],ixs_diags[jx]] + Psi_new;

    #log_max_diags = np.array([
    #  np.log(np.max(np.diag(Sig_mats[jx]).real) )
    #  for jx in range(d_dim)]);

    log_max_diags = np.array([
      np.log(np.min(Psi_diags[jx]) )
      for jx in range(d_dim)]);

    scale_factrs = np.exp(np.mean(log_max_diags) - log_max_diags);

    for jx in range(d_dim):
      Lam_mats[jx][:] = np.sqrt(scale_factrs[jx]) * Lam_mats[jx];
      Psi_diags[jx][:] = scale_factrs[jx] * Psi_diags[jx];
      Sig_mats[jx][:] = scale_factrs[jx] * Sig_mats[jx];

    max_norm_diff_vals[rx] = np.max([
      np.exp(
        np.log(np.linalg.norm(Sig_mats[jx] - Sig_mats_prev[jx], "fro") ) -
        np.log(np.linalg.norm(Sig_mats[jx], "fro") ))
      for jx in range(d_dim)]);

    for jx in range(d_dim):
      Sig_mats_prev[jx][:] = Sig_mats[jx];

    if np.isnan(max_norm_diff_vals[rx]) or (max_norm_diff_vals[rx] < conv_thrsh):
      break;

  try:
    ll = calculate_likelihood_array(X, Sig_mats);
  except:
    raise SSFA_RuntimeError;

  n_tot_obs = np.prod(X.shape);
  n_nonzero = np.array([np.sum(np.abs(Lam_mats[jx]) > 0) + p_dims[jx] for jx in range(d_dim)]);
  n_tot_models = p_dims * (k_dims + 1);
  log_n_choose_k = [(
    log_n_factorial(n_tot_models[jx]) -
    log_n_factorial(n_tot_models[jx] - n_nonzero[jx]) -
    log_n_factorial(n_nonzero[jx]) )
    for jx in range(d_dim)];

  ebic = -2 * ll + np.sum(n_nonzero) * np.log(n_tot_obs) + 2 * np.sum(log_n_choose_k);

  return {
    "Lam": Lam_mats,
    "Psi": Psi_diags,
    "ll": ll,
    "ebic": ebic,
    "max_norm_diff": max_norm_diff_vals[0:(rx + 1)]};


def ssfa(
  X: np.ndarray,
  k_dims: np.ndarray | int | None =None,
  Lam_init: np.ndarray | list | None =None,
  Psi_init: np.ndarray | list | None =None,
  adapt_penalty: np.ndarray | list | None =None,
  rho: np.float32 =0.0,
  n_reps: int =1000,
  conv_thrsh: np.float32 =1e-9,
  verbose: bool =False):

  """
  Description:
    Compute the sparse separable complex factor model for either array or vector variate data.
    This is just a wrapper function for the 'ssfa_vector()' and 'ssfa_array()' functions and
    will call the correct one based on the structure of the 'X' data.

  Arguments:
    |
    |------------- X: In the case of array-variate data, this is a (D + 1)-mode tensor of
    |                 dimension n x P1 x P2 x ... x PD, where n denotes the number of
    |                 observations, D denotes the number of modes of each array observation,
    |                 P1 is the dimension of the first mode, P2 is the dimension of the second
    |                 mode, so on and so forth. In the case of vector-variate data, this is an
    |                 n x P matrix.
    |
    |-------- k_dims: OPTIONAL. If set and the 'X' data is array-variate, this must be a numpy
    |                 vector with D integers denoting the number of factors to estimate at each
    |                 mode, where D denotes the number of modes of each array observation in 'X'.
    |                 If set and the 'X' data is vector-variate, this must either be an integer
    |                 or a numpy vector containing a single integer denoting the number of
    |                 factors to estimate.
    |
    |------ Lam_init: OPTIONAL. Initial factor loadings matrices. If set and the 'X' data is
    |                 array-variate, this must be a list with D elements, each of which must be
    |                 a numpy matrix. The number of columns of each matrix in this list will
    |                 determine the number of factors estimated for each mode. If set and the
    |                 'X' data is vector-variate, this is either a numpy matrix or a list with
    |                 just a single numpy matrix containing the initial factor loadings. This
    |                 argument will override the 'k_dims' argument.
    |
    |------ Psi_init: OPTIONAL. Initial diagonal terms of unique variance matrices. If set and
    |                 the 'X' data is array-variate, this must be a list with D elements, each
    |                 of which must be a Pj-dimensional numpy vector where Pj denotes the
    |                 dimension of the jth mode of the array observations in 'X'. If set and
    |                 the 'X' data is vector-variate, this is either a P-dimensional numpy
    |                 vector or a list with a single P-dimensional numpy vector of the initial
    |                 unique variances where P denotes the dimension of fthe vector observations
    |                 provided in the 'X' matrix.
    |
    |- adapt_penalty: OPTIONAL. Adaptive lasso penalty weights for factor loadings matrices.
    |                 If set and the 'X' data is array-variate, this must be a list with D
    |                 elements, each of which must be a numpy matrix. If 'Lam_init' is not set,
    |                 the number of columns of each matrix in this list will determine the
    |                 number of factors estimated for each mode. If 'Lam_init' is set, the
    |                 dimensions of each matrix must equal the dimensions of the corresponding
    |                 matrices in 'Lam_init'. If set and the 'X' data is vector-variate, this is
    |                 either a numpy matrix or a list with just a single numpy matrix
    |                 corresponding to the single factor loadings matrix. This argument will
    |                 override the 'k_dims' argument.
    |
    |----------- rho: OPTIONAL. Floating point sparsity shrinkage parameter. Default is 0.0;
    |
    |-------- n_reps: OPTIONAL. Integer denoting the max number of iterations to run the EM
    |                 algorithm. Default is 1000;
    |
    |---- conv_thrsh: OPTIONAL. Floating point value used to determine convergence of the EM
    |                 algorithm. Default is 1e-9.
    |
    |------- verbose: OPTIONAL. Boolean indicating whether or not to print algorithm runtime
    |                 details. Default is False.

  Return values:
    |
    |----------- Lam: List of D factor loadings matrices corresponding to each mode of the
    |                 data.
    |
    |----------- Psi: List of D numpy vectors providing the unique variances corresponding to
    |                 each mode of the data.
    |
    |------------ ll: The log-likelihood under the SSFA model based on the final model
    |                 estimates.
    |
    |---------- ebic: The extended BIC metric based on the final model estimates and the
    |                 log-likelihood.
    |
    |- max_norm_diff: Returned only for array-variate data. A vector providing the max
    |                 relative norm of the difference between one iteration's estimates of
    |                 each mode's covariance structures and the previous iteration's
    |                 estimates of each mode's covariance structures.
    |
    |----- norm_diff: Returned only for vector-variate data. A vector providing the
    |                 relative norm of the difference between one iteration's estimate of
    |                 the covariance structure and the previous iteration's estimate of
    |                 the covariance structure.
  """

  if not isinstance(X, np.ndarray):
    err_msg = "The provided variable 'X' must be a numpy array.";
    raise TypeError(err_msg);

  if len(X.shape) > 2:
    return ssfa_array(
      X,
      k_dims,
      Lam_init,
      Psi_init,
      adapt_penalty,
      rho,
      n_reps,
      conv_thrsh,
      verbose);

  else:
    return ssfa_vector(
      X,
      k_dims,
      Lam_init,
      Psi_init,
      adapt_penalty,
      rho,
      n_reps,
      conv_thrsh,
      verbose);


def rcnorm(n):
  """
  Generate n standard complex normal random variables

  Parameters:
  n (int): Number of samples to generate.

  Returns:
  numpy.ndarray: n x 1 array of standard  complex normal samples.
  """

  return (np.sqrt(0.5) * (np.random.standard_normal(n) + 1j * np.random.standard_normal(n) ));


def tucker_prod_seq(X: np.ndarray, mats: list):
  """
  Compute simple sequential Tucker product between array 'X' and the provided set of matrices
  provided by 'mats'. The first matrix is multiplied to the first mode of 'X', and each subsequent
  matrix is multiplied to the next mode of 'X' sequentially. 'X' is permuted along the way by
  cycling its dimensions to the left with wraparound, e.g. if the provided array 'X' has dimensions
  (p1 x p2 x p3 x p4) and two matrices are stored in the 'mats' list with dimensions (q1 x p1) and
  (q2 x p2) respectively, then the resulting array will have dimension (q2 x p3 x p4 x q1).
  Similarly, if three matrices are stored in the 'mats' list with dimensions (q1 x p1), (q2 x p2)
  and (q3 x p3) respectively, then the resulting array will have dimension q3 x p4 x q1 x q2.
  """

  assert isinstance(X, np.ndarray);
  assert isinstance(mats, list);

  X = X.copy();
  n_modes = len(X.shape);
  n_mats = len(mats);

  assert (n_mats > 0) & (n_mats <= n_modes);
  assert np.all([mats[mx].shape[1] == X.shape[mx] for mx in range(n_mats)]);

  dim_reshape_arr = list(X.shape);
  dim_reshape_arr[0] = mats[0].shape[0];
  X = (mats[0] @ X.reshape((X.shape[0], -1) )).reshape(dim_reshape_arr);

  if n_mats > 1:
    ixs_transpose = [*list(range(1, n_modes) ), 0];

    for mx in range(1, n_mats):
      X = X.transpose(ixs_transpose);
      dim_reshape_arr = list(X.shape);
      dim_reshape_arr[0] = mats[mx].shape[0];
      X = (mats[mx] @ X.reshape((X.shape[0], -1) )).reshape(dim_reshape_arr);

  return X;


def tucker_diag_prod_seq(X: np.ndarray, diags: list):
  """
  This function does the same thing as 'tucker_prod_seq()' except the matrices that get multiplied
  to each mode are diagonal matrices whose diagonal elements are stored in the 'diags' list.
  """

  assert isinstance(X, np.ndarray);
  assert isinstance(diags, list);

  X = X.copy();
  n_modes = len(X.shape);
  n_diags = len(diags);

  assert (n_diags > 0) & (n_diags <= n_modes);
  assert np.all([len(diags[dx]) == X.shape[dx] for dx in range(n_diags)]);

  dim_reshape_arr = list(X.shape);
  n_no_dx = np.prod(X.shape[1:]);
  diag_mat = diags[0].reshape((-1, 1) ).repeat(n_no_dx, 1);
  X = (diag_mat * X.reshape((X.shape[0], -1) )).reshape(dim_reshape_arr);

  if n_diags > 1:
    ixs_transpose = [*list(range(1, n_modes) ), 0];

    for dx in range(1, n_diags):
      X = X.transpose(ixs_transpose);
      dim_reshape_arr = list(X.shape);
      n_no_dx = np.prod(X.shape[1:]);
      diag_mat = diags[dx].reshape((-1, 1) ).repeat(n_no_dx, 1);
      X = (diag_mat * X.reshape((X.shape[0], -1) )).reshape(dim_reshape_arr);

  return X;


def pxem_w_lasso(XXt, n_no_jx, Lam, Psi, adapt_pen_mat, rho):
  p_dim, k_dim = Lam.shape;
  Psi_inv_Lam = (1 / Psi).reshape((-1, 1) ).repeat(k_dim, 1) * Lam;
  I_kjx = np.eye(k_dim, dtype=np.complex_);
  sher_wood_inv = np.linalg.inv(I_kjx + Lam.conj().T @ Psi_inv_Lam);
  sher_wood_Lam_Psi_inv = sher_wood_inv @ Psi_inv_Lam.conj().T;
  cov_sher_wood_chol = np.linalg.inv(
    np.linalg.cholesky(
      n_no_jx * sher_wood_inv + sher_wood_Lam_Psi_inv @ XXt @ sher_wood_Lam_Psi_inv.conj().T) );

  A_mat = (1 / np.sqrt(n_no_jx) ) * XXt @ sher_wood_Lam_Psi_inv.conj().T @ cov_sher_wood_chol.conj().T;
  A_mat_phase = np.angle(A_mat);
  A_mat_mod = np.abs(A_mat);
  Lam_new_mod = A_mat_mod - (Psi.reshape((-1, 1) ).repeat(k_dim, 1) * adapt_pen_mat * (rho / 2) );
  Lam_new_mod[Lam_new_mod < 0] = 0;
  Lam_new = Lam_new_mod * np.exp(A_mat_phase * 1j);
  Psi_new = ((1 / n_no_jx) * np.diag(XXt) - (Lam_new * Lam_new.conj() ).sum(axis=1) ).real;

  return (Lam_new, Psi_new);


def calculate_likelihood_vector(
  X: np.ndarray,
  Lam_mat: np.ndarray,
  Psi_diag: np.ndarray):
  """
  Description:
    Efficient computation of the log-likelihood of X and factor terms Lambda and Psi.

  Arguments:
    |
    |-------- X: Data matrix of dimension n x P, where n denotes the number of observations,
    |            and P denotes the dimension of each vector observation.
    |
    |-- Lam_mat: Factor loadings matrix of dimension P x k, where P denotes the dimension of
    |            each vector observation in 'X', and k denotes the number of factor loadings.
    |
    |- Psi_diag: P-dimensional vector of diagonal unique variance terms.
    |

  Return values:
    |
    |- Log-likelihood of model with vector-variate observations and factor analytic covariance
    |  structure.
    |
  """

  n_dim, p_dim = X.shape;
  p_dim_check, k_dim = Lam_mat.shape;

  if p_dim_check != p_dim:
    err_msg = "The matrix 'Lam_mat' must have a number of rows equal to the number of columns of X.";
    raise AssertionError(err_msg);

  Psi_inv_Lam = (1 / Psi_diag).reshape((-1, 1) ).repeat(k_dim, 1) * Lam_mat;
  I_k = np.eye(k_dim, dtype=np.complex_);
  sher_wood_chol = np.linalg.cholesky(I_k + Lam_mat.conj().T @ Psi_inv_Lam);
  Z_mat = np.linalg.inv(sher_wood_chol) @ Psi_inv_Lam.conj().T @ X.T;

  log_det_psi = n_dim * np.sum(np.log(Psi_diag) );
  log_det_sher_wood = 2 * n_dim * np.sum(np.log(np.diag(sher_wood_chol) )).real;
  tr_Xstar_Psi_inv_X = np.sum(X * X.conj() * (1 / Psi_diag).reshape((-1, 1) ).repeat(n_dim, 1).T).real;
  tr_Zstar_Z = np.sum(Z_mat * Z_mat.conj() ).real;

  return -(
    n_dim * p_dim * np.log(np.pi) +
    log_det_psi +
    log_det_sher_wood +
    tr_Xstar_Psi_inv_X -
    tr_Zstar_Z);


def calculate_likelihood_array(X: np.ndarray, Sig_mats: list):
  X = X.copy();

  n_dim = X.shape[0];
  p_dims = np.array(X.shape[1:]);
  d_dim = len(p_dims);
  n_modes = len(X.shape);

  L_inv_mats = [];
  log_dets = np.empty(d_dim);
  log_dets[:] = np.nan;

  for jx in range(d_dim):
    Sig_chol_jx = np.linalg.cholesky(Sig_mats[jx]);
    ixs_no_jx = np.array(sorted(set(range(d_dim) ).difference({jx}) ));
    log_dets[jx] = (
      n_dim * np.prod(p_dims[ixs_no_jx]) *
      2 * np.sum(np.log(np.diag(Sig_chol_jx) )).real);

    L_inv_mats.append(np.linalg.inv(Sig_chol_jx) );

  X = tucker_prod_seq(X.transpose([*list(range(1, n_modes) ), 0]), L_inv_mats);

  return -(
    n_dim * np.prod(p_dims) * np.log(np.pi) +
    np.sum(log_dets) +
    np.sum(X * X.conj() ).real);


def get_Psi_max(X):
  """
  Description:
    Estimate the unique variance components of a separable factor model where the factor loadings
    are all set to zero. This function is used to automatically find a grid of rho paramteters
    to use in the 'ssfa_param_sweep()' function.

  Arguments:
    |
    |- X: In the case of array-variate data, this is a (D + 1)-mode tensor of dimension
    |     n x P1 x P2 x ... x PD, where n denotes the number of observations, D denotes the
    |     number of modes of each array observation, P1 is the dimension of the first mode,
    |     P2 is the dimension of the second mode, so on and so forth. In the case of vector-
    |     variate data, this is an n x P matrix.
    |

  Return values:
    |
    |- List with D elements, each of which is the vector of estimated unique variances for the
    |  Dth mode of the dataset 'X' under a model where the factor loadings are assumed to be
    |  zero.
    |
  """

  if not isinstance(X, np.ndarray):
    err_msg = "The provided variable 'X' must be a numpy array.";
    raise TypeError(err_msg);

  n_modes = len(X.shape);

  if n_modes < 2:
    err_msg = "The provided variable 'X' must be either a numpy matrix or array.";
    raise AssertionError(err_msg);

  n_dim = X.shape[0];
  p_dims = np.array(X.shape[1:]);
  d_dim = len(p_dims);

  Psi_max_diags = [
    np.array([np.nan for ix in range(p_dims[jx])])
    for jx in range(d_dim)];

  for jx in range(d_dim):
    if d_dim > 1:
      ixs_no_jx = np.array(sorted(set(range(d_dim) ).difference({jx}) ));
      ixs_transpose = np.array([jx + 1, 0, *(ixs_no_jx + 1)]);
      X_jx = X.transpose(ixs_transpose).reshape((p_dims[jx], -1), order="F");
    else:
      X_jx = X.T;

    n_no_jx = X_jx.shape[1];
    Psi_max_diags[jx][:] = (1 / n_no_jx) * (X_jx * X_jx.conj() ).sum(axis=1).real;

  if d_dim > 1:
    ixs_transpose_left = [*list(range(1, n_modes) ), 0];

    n_reps = 10;

    #blah_prev = [Psi_max_diags[jx].copy() for jx in range(d_dim)];
    #blah = np.array([np.nan for rx in range(n_reps)]);

    for rx in range(n_reps):
      for jx in range(d_dim):
        ixs_no_jx = np.array(sorted(set(range(d_dim) ).difference({jx}) ));
        ixs_transpose = np.array([*(ixs_no_jx + 1), jx + 1, 0]);

        diags_sqrt_inv_jx = [1 / np.sqrt(Psi_max_diags[njx]) for njx in ixs_no_jx];
        X_jx = (
          tucker_diag_prod_seq(X.transpose(ixs_transpose), diags_sqrt_inv_jx).\
          transpose(ixs_transpose_left) );

        X_jx = X_jx.reshape((X_jx.shape[0], -1) );
        n_no_jx = X_jx.shape[1];
        Psi_max_diags[jx][:] = (1 / n_no_jx) * (X_jx * X_jx.conj() ).sum(axis=1).real;

      log_max_diags = np.array([
        np.log(np.min(Psi_max_diags[jx]) )
        for jx in range(d_dim)]);

      scale_factrs = np.exp(np.mean(log_max_diags) - log_max_diags);

      for jx in range(d_dim):
        Psi_max_diags[jx][:] = scale_factrs[jx] * Psi_max_diags[jx];

      #blah[rx] = np.mean((np.concatenate(Psi_max_diags) - np.concatenate(blah_prev) )**2);
      #blah_prev = [Psi_max_diags[jx].copy() for jx in range(d_dim)];

  return Psi_max_diags;


def log_n_factorial(n):
  # This uses the Ramanujan approximation for computing log(n!) when n is small,
  # and the Stirling approximation when n is large.

  if n < 0:
    raise AssertionError("The provided variable 'n' must be non-negative.");

  if n == 0:
    log_fact = 0.0;

  elif n <= 10:
    log_fact = np.sum(np.log(np.arange(1, n + 1) ));

  elif n <= 1000:
    log_fact = (
      n * (np.log(n) - 1) +
      (np.log(n) + np.log(1 + 4 * n * (1 + 2 * n) )) / 6 +
      np.log(np.pi) / 2);

  else:
    adj = np.log(np.pi) / 2;
    log_fact = (n + adj) * (np.log(n + adj) - 1);

  return log_fact;


class SSFA_RuntimeError(Exception):
  def __init__(self, *args, **kwargs):
    err_msg = \
      "Failed to converge. Consider reducing the number of factors\n" +\
      "or try new initial values.";

    super().__init__(err_msg, *args, **kwargs);
