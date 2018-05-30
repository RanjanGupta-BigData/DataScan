from scipy.fftpack import fftn, fftshift, fftfreq
import xarray as xr
import numpy as np
from scipy.signal import hilbert, get_window, medfilt
from scipy.interpolate import griddata, InterpolatedUnivariateSpline as ius
from collections import OrderedDict
import itertools


class DataScan(xr.DataArray):
    def fft(self, shape=None, dims=None, axes=None, ssb=False):
        """
        Computes the Discrete Fourier transform along a specified dimension or axis. The FFT is
        computed along the requested axis only if the axis labels are uniformly sampled.

        Parameters
        ----------
        shape : int, array_like, optional
            The size of the fft. If not specified, the current size of the axis is taken.

        dims : str, array_like, optional
            A string or array of strings giving the names of the dimensions along which to
            compute the Fourier transform.

        axes : int, array_like, optional
            The axes numbers along which to compute the FFT. Note that only one of `axes` and
            `dims` can be specified.

        ssb : bool, optional
            Determines if only the single sided Fourier transform will be returned.

        Returns
        -------
        : DataScan
            A new signal representing the Fourier transform.

        Note
        ----
        Keyword arguments can be given to the the underlying Fourier transform function
        :func:`scipy.fftpack.fft2`.

        Note
        ----
        The returned FFT values are shifted so that the axes labels are monotonous, i.e. running
        from negative frequencies up to positive frequencies. This is different from the usual
        way an FFT solution is returned.
        """
        if dims is not None and axes is not None:
            raise ValueError('Cannot have both "dims" amd "axes" to be specified.')

        if dims is None and axes is None:
            axes = self.get_axis_num(self.dims)

        if dims is not None:
            axes = self.get_axis_num(dims)

        if not hasattr(axes, '__len__'):
            axes = [axes]

        if not hasattr(shape, '__len__') and shape is not None:
            shape = [shape]

        # get coordinates
        coords = list(self.indexes.values())

        # first check if the data is uniformly sampled across the requested axes
        for ax in axes:
            mean_fs = np.mean(np.diff(coords[ax]))
            if not np.allclose(np.diff(coords[ax], 2), 0.0, atol=mean_fs/5):
                raise ValueError('Axes used for FFT should be uniformed sampled.')

        Y = fftshift(fftn(self, shape=shape, axes=axes), axes=axes)

        # compute the axes values
        for ax in axes:
            coords[ax] = fftshift(fftfreq(len(coords[ax]) if shape is None else shape[ax],
                                          np.mean(np.diff(coords[ax]))))
        ds = DataScan(Y, coords=coords, dims=self.dims)

        # compute the single side band
        if ssb:
            ssb_coords = {self.dims[i]: coords[i][coords[i] >= 0] for i in axes}
            ds = ds.sel(**ssb_coords)

        return ds

    def __call__(self, values, dim, *args, **kwargs):
        """
        Re-indexes a specified dimension using interpolation.

        Parameters
        ----------
        values : array_like, slice
        """
        if dim not in self.dims:
            raise ValueError('Unknown dim specified.')

        ds = self.reindex(indexers={dim: values})
        ax = self.get_axis_num(dim)
        b = np.apply_along_axis(lambda x: ius(self.coords[dim], x, **kwargs)(values),
                                ax, np.array(self))
        ds.values = b
        return ds
        #print(ds)
        # if hasattr(key, '__len__'):
        #     return Signal(self._interp_fnc(key), index=key)

    def operate(self, opts, dim=None):
        """
        Returns the signal according to a given option.

        Parameters
        ----------
        opts : string/char, optional
            The possible options are (combined options are allowed):

             +--------------------+--------------------------------------+
             | *option*           | Meaning                              |
             +====================+======================================+
             | '' *(Default)*     | Return the raw signal                |
             +--------------------+--------------------------------------+
             | 'n'                | normalized signal                    |
             +--------------------+--------------------------------------+
             | 'd'                | decibel value                        |
             +--------------------+--------------------------------------+
             | 'e'                | compute the signal envelop           |
             +--------------------+--------------------------------------+

        dim : str, optional
            Only used in the case option specified 'e' for envelop. Specifies along which axis to
            compute the envelop. if not specified, it will take the first dimension by default.

        Returns
        -------
        : DataScan
            The modified DataScan.
        """
        yout = self

        if dim is not None:
            axis = self.get_axis_num(dim)
        else:
            axis = self.get_axis_num(self.dims[0])

        if 'e' in opts:
            # make hilbert transform faster by computing it at powers of 2
            n = self.shape[axis]
            n = 2**int(np.ceil(np.log2(n)))
            yout = np.abs(hilbert(yout.values, N=n, axis=axis))
            yout = yout[:self.shape[0], :self.shape[1]]
        if 'n' in opts:
            # TODO: call normalize function
            yout = yout/np.abs(yout).max().max()
        if 'd' in opts:
            yout = 20*np.log10(np.abs(yout))
        return DataScan(yout, coords=self.coords)

    def peaks(self, dim, threshold=0.1, min_dist=None, by_envelop=False):
        """
        This is a 1D function that can be applied to a multidimensional array along a single
        dimension. Finds the peaks by taking its first order difference. By using *threshold* and
        *min_dist* parameters, it is possible to reduce the number of detected peaks.

        Parameters
        ----------
        threshold : float, [0., 1.]
            Normalized threshold. Only the peaks with amplitude higher than the
            threshold will be detected.

        min_dist : float
            The minimum distance in index units between ech detected peak. The peak with the highest
            amplitude is preferred to satisfy this constraint.

        by_envelop : bool
            Compute the peaks of the signal based on its envelop.

        Returns
        -------
        : ndarray
            Array containing the indexes of the peaks that were detected

        Notes
        -----
        This method is adapted from the peak detection method in
        [PeakUtils](http://pythonhosted.org/PeakUtils/)
        """
        if threshold > 1 or threshold <= 0:
            raise ValueError('Threshold should be in the range (0.0, 1.0].')

        if min_dist is None:
            min_dist = self.ts[dim]

        if min_dist <= 0.0:
            raise ValueError('min_dist should be a positive value.')

        axis = self.get_axis_num(dim)
        dims = list(self.dims)
        dims.remove(dim)
        dim_coord = self.coords[dim].values

        out = []
        # loop over all other dimensions
        for c in itertools.product(*[list(self.indexes[i]) for i in dims]):
            y = self.sel(**{d: c[i] for i, d in enumerate(dims)})
            y = y.operate('ne') if by_envelop else abs(y.operate('n'))
            dy = np.diff(y)
            peaks = np.where((np.hstack([dy, 0.]) < 0.)
                             & (np.hstack([0., dy]) > 0.)
                             & (y > threshold))[0]
            min_dist = int(min_dist/y.ts[dim])

            if peaks.size > 1 and min_dist > 1:
                highest = peaks[np.argsort(y[peaks].values)][::-1]
                rem = np.ones(y.size, dtype=bool)
                rem[peaks] = False

                for peak in highest:
                    if not rem[peak]:
                        sl = slice(max(0, peak - min_dist), peak + min_dist + 1)
                        rem[sl] = True
                        rem[peak] = False
                peaks = np.arange(y.size)[~rem]
            out.extend([c[:axis] + (dim_coord[pk],) + c[axis:] for pk in peaks])
        return out

    def window(self, dim=None, index1=None, index2=None, win_fcn='boxcar'):
        """
        Applies a window to the signal within a given time range. Currently only supports 1-D
        windows.

        Parameters
        ----------
        index1 : float or int, optional
            The start index/position of the window. Default value is minimum of index.

        index2 : float or int, optional
            The end index/position of the window. Default value is maximum of index.

        win_fcn : string/float/tuple, optional
            The type of window to create. See the function
            :func:`scipy.signal.get_window()` for a complete list of
            available windows, and how to pass extra parameters for a
            specific window function.

        Returns
        -------
        Signal:
            The windowed Signal signal.

        Note
        ----
          If the window requires no parameters, then `win_fcn` can be a string.
          If the window requires parameters, then `win_fcn` must be a tuple
          with the first argument the string name of the window, and the next
          arguments the needed parameters. If `win_fcn` is a floating point
          number, it is interpreted as the beta parameter of the kaiser window.
        """
        if self.ndim == 1 and dim is None:
            dim = self.dims[0]

        out = self.copy()
        axis = self.get_axis_num(dim)
        axis_len = self.shape[axis]
        wind = DataScan(np.zeros(axis_len), coords=[(dim, self.coords[dim])])
        wind.loc[{dim: slice(index1, index2)}] = get_window(win_fcn,
                                                            len(wind.loc[{dim: slice(index1,
                                                                                     index2)}]))
        b = np.apply_along_axis(lambda x: x*wind, axis, self)
        out.values = b
        return out

    def limits(self, threshold=None):
        """
        Computes the index limits where the signal first goes above a given threshold,
        and where it last goes below this threshold. Linear interpolation is used to find the
        exact point for this crossing.

        Parameters
        ----------
        threshold : float, optional, (0.0, 1.0]
            Normalized value where the signal first rises above and last falls below. If no value is
            specified, the default is the root mean square of the signal.

        Returns
        -------
        start_index, end_index (tuple (2,)):
            A two element tuple representing the *start_index* and *end_index* of the signal.
        """
        if self.ndim > 1:
            raise ValueError('This method supports only 1-D arrays.')
        coord = self.coords[self.dims[0]].values

        if threshold is None:
            threshold = self.std().values
        if threshold <= 0 or threshold > 1:
            raise ValueError("`threshold` should be in the normalized (0.0, 1.0].")

        senv = self.operate('n').values
        ind = np.where(senv >= threshold)[0]
        return coord[ind[0]], coord[ind[-1]]

    def bandwidth(self, threshold=None, nfft=None, smoothing=False, win_len=None):
        """
        Computes the bandwidth of the signal by finding the range of frequencies
        where the signal is above a given threshold.

        Parameters
        ----------
        threshold : float, optional, (0.0, 1.0]
            The normalized threshold for which to compute the bandwidth. If this is not
            specified, the threshold is set to the root mean square value of the signal.

        nfft : bool, optional
            Since this computation is based on the Fourier transform, indicate the number of
            points to be used on computing the FFT.

        Returns
        -------
        : float
            The total signal bandwidth.
        """
        if self.ndim > 1:
            raise ValueError('This method is supported only for 1-D arrays.')

        if threshold <= 0 or threshold > 1:
            raise ValueError("Threshold should be in the range (0.0, 1.0].")

        fdomain = abs(self.fft(ssb=True, shape=nfft))

        if smoothing:
            if win_len is None:
                # use 2% the size of the signal
                win_len = int(0.02*len(fdomain))
                win_len = win_len + 1 if win_len % 2 == 0 else win_len
            fdomain.values = medfilt(fdomain, win_len)

        lims = fdomain.limits(threshold)
        return lims[1] - lims[0]

    def frequency(self, option='center', threshold=None, nfft=None, smoothing=False, win_len=None):
        """
        Computes the center or peak frequency of the signal. Only for 1-D arrays.

        Parameters
        ----------
        option : str, {'center', 'peak'}
            Specify whether to compute the center or peak frequency of the signal.

        threshold : float, optional, (0.0, 1.0]
            Threshold value indicating the noise floor level. Default value is the root mean
            square. Only required if option = 'center'.

        nfft : bool, optional
            Since this computation is based on the Fourier transform, indicate the number of
            points to be used on computing the FFT.

        Returns
        -------
        : float
            The value of the center frequency.
        """
        if self.ndim > 1:
            raise ValueError('This method supports only 1-D arrays.')

        if (threshold is not None) and (threshold <= 0 or threshold > 1):
            raise ValueError("Threshold should be in the range (0.0, 1.0].")

        fdomain = abs(self.fft(ssb=True, shape=nfft))

        if smoothing:
            if win_len is None:
                # use 2% the size of the signal
                win_len = int(0.02 * len(fdomain))
                win_len = win_len + 1 if win_len % 2 == 0 else win_len
            fdomain.values = medfilt(fdomain, win_len)

        if option == 'center':
            minf, maxf = fdomain.limits(threshold)
            return (minf+maxf)/2.0
        if option == 'peak':
            return self.coords[self.dims[0]][fdomain.argmax().values].values

        raise ValueError('`option` value given is unknown. Supported options: {"center", "peak"}.')

    def segment(self, threshold, pulse_width, dim=None, min_dist=None, holdoff=None,
                win_fcn='boxcar'):
        """
        Segments the signal into a collection of signals, with each item in the collection,
        representing the signal within a given time window. This is usually useful to
        automate the extraction of multiple resolvable echoes.

        Parameters
        ----------
        threshold : float
            A threshold value (in dB). Search for echoes will be only for signal values
            above this given threshold. Note that the maximum threshold is 0 dB, since
            the signal will be normalized by its maximum before searching for echoes.

        pulse_width : float
            The expected pulse_width. This should have the same units as the units of the Signal
            index. If this is not known exactly, it is generally better to specify this
            parameter to be slightly larger than the actual pulse_width.

        min_dist : float
            The minimum distance between the peaks of the segmented signals.

        holdoff : float, optional
            The minimum index for which to extract a segment from the signal.

        win_fcn : string, array_like
            The window type that will be used to window each extracted segment (or echo). See
            :func:`scipy.signal.get_window()` for a complete list of available windows,
            and how to pass extra parameters for a specific window type, if needed.

        Returns
        -------
            : list
            A list with elements of type :class:`Signal`. Each Signal element represents an
            extracted segment.
        """
        if self.ndim == 1 and dim is None:
            dim = self.dims[0]

        if self.ndim > 1 and dim is None:
            raise ValueError('dim must be specified if more than one dimension.')

        if holdoff is not None:
            y = self.loc[{dim: slice(holdoff)}]
        else:
            y = self.copy()

        if min_dist is None:
            min_dist = pulse_width

        pks = y.peaks(dim, threshold, min_dist)

        if len(pks) == 0:
            return xr.Dataset()
        axis = y.get_axis_num(dim)
        # remove segment if its end is over the limit of signal end
        pks = [pk for pk in pks if pk[axis] + pulse_width <= y.coords[dim][-1]]

        if self.ndim > 1:
            out = xr.Dataset()
            # TODO: add coordinate information to the returned dataset
            for i, pk in enumerate(pks):
                self_slice = self.loc[{d: pk[j] for j, d in enumerate(self.dims) if d != dim}]
                out['pk_' + str(i)] = self_slice.window(dim, index1=pk[axis]-pulse_width/2,
                                                        index2=pk[axis]+pulse_width/2,
                                                        win_fcn=win_fcn)
        else:
            # special handling for cases when signal is 1-D - return DataArray
            pks = [pk[0] for pk in pks]
            out = xr.DataArray(np.zeros((self.shape[axis], len(pks))),
                               coords=[(dim, self.coords[dim]), ('dim2', pks)])
            for p in pks:
                out.loc[{'dim2': p}] = y.window(dim, index1=p-pulse_width/2,
                                                index2=p+pulse_width/2,
                                                win_fcn=win_fcn)
        return out

    # def sparse_pse(self, dim, threshold, fc, pulse_width, overlap=0, nfft=None, win_fcn='boxcar'):
    #     """
    #     Computes the sparse power spectral estimate
    #
    #     Parameters
    #     ----------
    #     fc
    #     width
    #     overlap
    #     nfft
    #
    #     Returns
    #     -------
    #
    #     """
    #     echoes = self.segment(dim, threshold=threshold, pulse_width=pulse_width,
    #                           min_dist=pulse_width-overlap, win_fcn=win_fcn)
    #     out = 0
    #     for k in echoes.data_vars:
    #         Y = DataScan(echoes[k]).fft(dims=dim, shape=nfft, ssb=True)
    #         out += abs(Y).sel(**{dim: fc}, method='nearest')**2
    #     return out

    def skew(self, angle, dim, other_dim=None, **kwargs):
        """
        Applied a skew transformation on the data. Currently, we support skew along a single
        dimension only.

        Parameters
        ----------
        angle : float
            The angle to skew the coordinates in degrees.

        dim : integer, str
            The dimension along which to skew.

        other_dim : integer, str
            The second dimension which forms the plane for which to apply the skew.
            This argument is optional only for 2-dimensional arrays. Otherwise it should
            always be specified.
        """
        if other_dim is None and self.ndim > 2:
            raise ValueError('other_dim must be provided if ndim > 2.')

        if other_dim is None and self.ndim == 2:
            d = dict.fromkeys(self.dims)
            del d[dim]
            other_dim = list(d.keys())[0]
        axis = self.get_axis_num(dim)
        other_axis = self.get_axis_num(other_dim)
        skew_matrix = np.eye(self.ndim)
        skew_matrix[axis, other_axis] = np.tan(np.deg2rad(angle))
        return self._skew_by_matrix(skew_matrix, **kwargs)

    def cscan(self, theta=None, mode='max'):
        """
        Specialized method for computing the C-Scans from Ultrasound Testing raster scans. This
        collapses the time axis and provides a top view of the scan, along the *x-y* axes.

        Parameters
        ----------
        theta : float
            The angle for which to skew the scan. This should be the wave propagation angle. The
            c-scan will be the top view after skew by the given angle.

        Returns
        -------
        : ultron.DataScan
            The computed C-scan.
        """
        if theta is None:
            return self.max(dim='Z')

        dx = self.ts['X']

        # get the skewed coordinates at a single Y-coordinate
        bscan = self.isel(Y=0)
        bscan_dims = bscan.dims
        coords = self.isel(Y=0).skew(theta, 'X')
        coords['X'] /= dx
        coords['X'] = coords['X'].astype(np.int32)

        xmin, xmax = np.min(coords['X']), np.max(coords['X'])
        nx = xmax - xmin + 1

        out = np.zeros((self.sizes['Y'], nx))
        new_scan = abs(self)

        for i, xval in enumerate(range(xmin, xmax + 1)):
            # indx, indz = np.where(coords['X'] == xval)
            ind1, ind2 = np.where(coords['X'] == xval)
            # out[:, i] = scan.isel_points(X=indx, Z=indz).max('points')
            if mode == 'max':
                out[:, i] = new_scan.isel_points(**{bscan_dims[0]: ind1, bscan_dims[1]: ind2}).max(
                    'points')
            elif mode == 'std':
                out[:, i] = abs(new_scan.isel_points(**{bscan_dims[0]: ind1,
                                                    bscan_dims[1]: ind2})).std('points')
            else:
                raise ValueError('Unknown mode. Only std and max supported.')
        x = np.arange(xmin, xmax + 1) * dx
        return DataScan(out, coords=[new_scan.coords['Y'], x], dims=['Y', 'X'])

    def _skew_by_matrix(self, skew_matrix, interpolate=False, ts=None, **kwargs):
        """
        A general skew function on all coordinates. This requires knowledge of the skew
        coordinates, since the skew matrix must be provided.

        Parameters
        ----------
        skew_matrix: array_like
            The skew matrix used to skew the array. If the :class:`DataScan` has *n* dimensions,
            the the skew_matrix must be a *n x n* matrix.

        interpolate : bool, optional
            Specify weather or not to perform interpolation. If :const:`False`, then only the
            skewed coordinates are computed and returned.

        ts : scalar, array_like, optional
            Specific the sampling rate for the interpolation. If not specified, sampling rate is
            the same as the original axis. If a scalar is given, all the dimensions are sampled
            with the same rate. Only required if :obj:`interpolate` is :const:`True`.

        Returns
        -------
        : *skewed_dims
            If :obj:`
        """
        # get coordinates as a list
        coord_vals = [v.values for k, v in self.indexes.items()]
        arrays = np.array([v.ravel() for v in np.meshgrid(*coord_vals, indexing='ij')])

        # Apply the transformation matrix
        skewed_coords = np.dot(skew_matrix, arrays)

        # dictionary to hold output coordinates
        out = OrderedDict.fromkeys(self.dims)

        if interpolate:
            if ts is None:
                ts = list(self.ts.values())
            elif not hasattr(ts, '__len__'):
                ts = [ts]*self.ndim
            else:
                if len(ts) != self.ndim:
                    raise ValueError('ts must have size the same as number of dimensions,'
                                     'or it should just be a scalar.')

            for i, coord in enumerate(skewed_coords):
                out[self.dims[i]] = np.arange(np.min(coord), np.max(coord), ts[i])

            mesh_coords = np.meshgrid(*list(out.values()), indexing='ij')
            vals = griddata(tuple(skewed_coords), self.values.ravel(), tuple(mesh_coords), **kwargs)

            # TODO: fix borders (extrapolation) for nearest neighbor interpolation
            return DataScan(vals, coords=out, dims=self.dims)
        else:
            for k, coord in zip(out, skewed_coords):
                out[k] = coord.reshape(self.shape)
                out[k] = np.round(out[k]/self.ts[k]) * self.ts[k]
            return out

    @property
    def ts(self):
        """
        Compute the mean sampling interval for each coordinate.
        """
        out = OrderedDict.fromkeys(self.dims)
        for k in self.indexes:
            out[k] = np.mean(np.diff(self.coords[k]))
        return out

    @property
    def fs(self):
        """
        Compute the sampling frequency.
        """
        out = self.ts
        for k in out:
            out[k] = 1/out[k]
        return out
