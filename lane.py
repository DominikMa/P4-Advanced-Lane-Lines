"""Define a class to receive the characteristics of each line detection."""


import numpy as np

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720  # meters per pixel in y dimension
xm_per_pix = 3.7/880  # meters per pixel in x dimension


class Lane():
    """Class to represent a lane."""

    def __init__(self):
        """Set initial values."""
        # was the line detected in the last iteration?
        self.detected = False
        self.bad_Frames = 5
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients of the last n fits of the line
        self.recent_fitted = []
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #  polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #  x values of the curvature of the line in some units
        self.recent_radius_of_cur = []
        #  average x values of curvature of the line in some units
        self.best_radius_of_curvature = None
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

    def bad_Frame(self):
        """Count bad frames."""
        self.detected = False
        self.bad_Frames = self.bad_Frames + 1

    def checks(self, poly_fit, poly_fitx, poly_fity):
        """Check if lane makes sense."""
        if self.best_fit is not None:
            if not (np.abs(self.best_fit-poly_fit) <
                    np.array([0.001, 1, 500])).all():
                return False
        if self.bestx is not None:
            if np.mean(np.abs(self.bestx-poly_fitx)) > 200:
                return False

        return True

    def update(self, x, y, shape, smooth=7):
        """Update the lane."""
        if self.bad_Frames > 0:
            self.bad_Frames = self.bad_Frames - 1
        self.detected = True
        # Fit a second order polynomial to each
        poly_fit = np.polyfit(y, x, 2)

        # Generate x and y values
        poly_fity = np.linspace(0, shape[0]-1, shape[0])

        try:
            poly_fitx = (poly_fit[0]*poly_fity**2 +
                         poly_fit[1]*poly_fity +
                         poly_fit[2])
            self.checks(poly_fit, poly_fitx, poly_fity)
        except TypeError:
            # Avoids an error if `poly_fitx` still none or incorrect
            print('The function failed to fit a line!')
            self.detected = False

        if self.detected:
            if len(self.recent_xfitted) > smooth:
                self.recent_xfitted = self.recent_xfitted[1:]
            self.recent_xfitted.append(poly_fitx)
            if self.bestx is not None:
                self.diffs = self.bestx - np.mean(self.recent_xfitted)
            self.bestx = np.mean(self.recent_xfitted, axis=0)

            if len(self.recent_fitted) > smooth:
                self.recent_fitted = self.recent_fitted[1:]
            self.recent_fitted.append(poly_fit)
            self.best_fit = np.mean(self.recent_fitted, axis=0)

            self.current_fit = poly_fit

            y_eval = np.max(poly_fity)
            poly_fit_meter = np.polyfit(poly_fity*ym_per_pix,
                                        poly_fitx*xm_per_pix, 2)
            new_curvature = (((1 + (2*poly_fit_meter[0]*y_eval*ym_per_pix +
                                    poly_fit_meter[1])**2)**1.5) /
                             np.absolute(2*poly_fit_meter[0]))

            if len(self.recent_radius_of_cur) > smooth:
                self.recent_radius_of_cur = self.recent_radius_of_cur[1:]
            self.recent_radius_of_cur.append(new_curvature)
            self.best_radius_of_curvature = np.mean(self.recent_radius_of_cur)
            self.radius_of_curvature = new_curvature

            self.line_base_pos = np.mean(self.bestx[-20])

            self.allx = x
            self.ally = y
