# Streamlit Community Cloud Deployment Guide

## 🎯 Quick Overview
Your WPI dashboard is now ready for deployment! Here's what we've created:

- ✅ `streamlit_app.py` - Main Streamlit application
- ✅ `requirements_streamlit.txt` - Dependencies for Streamlit Cloud
- ✅ `wpi_10_commodities.csv` - Your data file

## 📋 Pre-deployment Checklist

### 1. GitHub Repository Setup
```bash
# If you haven't already, initialize git repo
git init
git add .
git commit -m "Initial WPI Streamlit dashboard"

# Create GitHub repository and push
git remote add origin https://github.com/yourusername/wpi-dashboard
git push -u origin main
```

### 2. File Structure (should look like this):
```
your-repo/
├── streamlit_app.py           # Main app
├── requirements_streamlit.txt  # Dependencies  
├── wpi_10_commodities.csv     # Data file
└── README.md                  # Optional
```

## 🚀 Deployment Steps

### Step 1: Access Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account

### Step 2: Deploy New App
1. Click **"New app"**
2. Select your GitHub repository
3. Set these parameters:
   - **Branch**: `main` (or your default branch)
   - **Main file path**: `streamlit_app.py`
   - **App URL**: Choose a custom name like `wpi-india-dashboard`

### Step 3: Configure (if needed)
- Streamlit will automatically use `requirements_streamlit.txt`
- No additional environment variables needed
- Data file (`wpi_10_commodities.csv`) will be loaded automatically

### Step 4: Deploy!
- Click **"Deploy!"**
- Wait 2-3 minutes for build to complete
- Your app will be live at `https://your-app-name.streamlit.app`

## 🔧 Key Differences from Dash

| Dash Version | Streamlit Version |
|--------------|-------------------|
| `app.callback()` decorators | Direct widget updates |
| `dcc.Dropdown()` | `st.multiselect()` |
| `dcc.RangeSlider()` | `st.slider()` |
| `html.Div()` layout | `st.columns()` layout |
| `app.run()` server | `streamlit run` command |

## 📊 Features Preserved
- ✅ All 4 interactive charts
- ✅ Commodity selection dropdown
- ✅ Year range slider  
- ✅ Real-time updates
- ✅ Summary statistics
- ✅ Professional styling
- ✅ Responsive layout

## 🐛 Troubleshooting

### Build Fails?
- Check `requirements_streamlit.txt` has correct versions
- Ensure `wpi_10_commodities.csv` is in the same directory
- Verify file paths are correct in the code

### Data Not Loading?
- Confirm CSV file is committed to GitHub
- Check file name matches exactly: `wpi_10_commodities.csv`

### App Crashes?
- Check Streamlit logs in the deployment interface
- Verify all required columns exist in your CSV

## 🎨 Customization Options

### Change Theme
Add to your repo root as `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"  
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

### Custom Domain (Pro Feature)
- Available with Streamlit Cloud Pro
- Connect your custom domain in settings

## 📈 Next Steps After Deployment

1. **Share your dashboard**: Get the public URL and share it
2. **Monitor usage**: Check visitor stats in Streamlit Cloud  
3. **Update data**: Push new CSV files to auto-update the app
4. **Add features**: More visualizations, filters, or data sources

## 🔗 Useful Links
- [Streamlit Documentation](https://docs.streamlit.io)
- [Streamlit Community Cloud](https://share.streamlit.io)
- [Plotly Documentation](https://plotly.com/python/)

---

**Ready to deploy? Run this locally first:**
```bash
streamlit run streamlit_app.py
```

Then follow the deployment steps above! 🚀